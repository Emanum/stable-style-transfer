# ---
# output-directory: "/tmp/stable-diffusion-xl"
# args: ["--prompt", "An astronaut riding a green horse"]
# runtimes: ["runc", "gvisor"]
# ---
# # Stable Diffusion XL 1.0
#
# This example is similar to the [Stable Diffusion CLI](/docs/examples/stable_diffusion_cli)
# example, but it generates images from the larger SDXL 1.0 model. Specifically, it runs the
# first set of steps with the base model, followed by the refiner model.
#
# [Try out the live demo here!](https://modal-labs--stable-diffusion-xl-app.modal.run/) The first
# generation may include a cold-start, which takes around 20 seconds. The inference speed depends on the GPU
# and step count (for reference, an A100 runs 40 steps in 8 seconds).

# ## Basic setup

from pathlib import Path

from modal import Image, Mount, Stub, asgi_app, gpu, method


# ## Define a container image
#
# To take advantage of Modal's blazing fast cold-start times, we'll need to download our model weights
# inside our container image with a download function. We ignore binaries, ONNX weights and 32-bit weights.
#
# Tip: avoid using global variables in this function to ensure the download step detects model changes and
# triggers a rebuild.


def download_models():
    from huggingface_hub import snapshot_download

    ignore = ["*.bin", "*.onnx_data", "*/diffusion_pytorch_model.safetensors"]
    snapshot_download(
        "stabilityai/stable-diffusion-xl-base-1.0", ignore_patterns=ignore
    )
    snapshot_download(
        "lllyasviel/sd-controlnet-canny", ignore_patterns=ignore
    )
    snapshot_download(
        "lllyasviel/control_v11f1p_sd15_depth", ignore_patterns=ignore
    )


image = (
    Image.debian_slim()
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "diffusers~=0.19",
        "invisible_watermark~=0.1",
        "transformers~=4.31",
        "accelerate~=0.21",
        "safetensors~=0.3",
        "accelerate",
        "opencv-python",
        "numpy",
        "omegaconf",
        "pillow"
    )
    .run_function(download_models)
)

stub = Stub("stable-style-transfer", image=image)

with stub.image.run_inside():
    import numpy as np
    import cv2
    from PIL import Image
    from diffusers.utils import load_image
    import torch


# ## Load model and run inference
#
# The container lifecycle [`__enter__` function](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta)
# loads the model at startup. Then, we evaluate it in the `run_inference` function.
#
# To avoid excessive cold-starts, we set the idle timeout to 240 seconds, meaning once a GPU has loaded the model it will stay
# online for 4 minutes before spinning down. This can be adjusted for cost/experience trade-offs.


@stub.cls(gpu=gpu.A10G(), container_idle_timeout=240)
class Model:
    def __enter__(self):
        import torch
        from diffusers import DiffusionPipeline, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, \
            UniPCMultistepScheduler
        from transformers import pipeline

        load_options = dict(
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
        )

        # Load base model

        # Load ControlNet
        self.canny_controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",
                                                                torch_dtype=torch.float16,
                                                                use_safetensors=True)

        self.depth_controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth",
                                                                torch_dtype=torch.float16,
                                                                use_safetensors=True)

        self.depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", controlnet=[self.depth_controlnet, self.canny_controlnet],
            torch_dtype=torch.float16,
            use_safetensors=True).to("cuda")

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()

        # Compiling the model graph is JIT so this will increase inference time for the first run
        # but speed up subsequent runs. Uncomment to enable.
        # self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        # self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)

    @method()
    def inference(self, prompt):
        # negative_prompt = "disfigured, ugly, deformed"

        # load image from url
        init_image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

        # calc canny
        canny_img = np.array(init_image)
        canny_img = cv2.Canny(canny_img, 100, 200)
        canny_img = canny_img[:, :, None]
        canny_img = np.concatenate([canny_img, canny_img, canny_img], axis=2)
        canny_image = Image.fromarray(canny_img)

        # calc depth_map
        # depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")  # dpt-hybrid-midas
        depth_map = Model().get_depth_map.remote(init_image, self.depth_estimator).unsqueeze(0).half().to("cuda")
        # depth_map = self.get_depth_map(init_image, self.depth_estimator).unsqueeze(0).half().to("cuda")

        image = self.pipe(
            prompt, image=[init_image], control_image=[depth_map, canny_image],
        ).images[0]
        # image = canny_image

        import io

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return image_bytes

    @method()
    def get_depth_map(self, image, depth_estimator):
        image = depth_estimator(image)["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        detected_map = torch.from_numpy(image).float() / 255.0
        depth_map = detected_map.permute(2, 0, 1)
        return depth_map


# And this is our entrypoint; where the CLI is invoked. Explore CLI options
# with: `modal run stable_diffusion_xl.py --prompt 'An astronaut riding a green horse'`


@stub.local_entrypoint()
def main(prompt: str):
    image_bytes = Model().inference.remote(prompt)

    dir = Path("./stable-diffusion-xl")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)


# ## A user interface
#
# Here we ship a simple web application that exposes a front-end (written in Alpine.js) for
# our backend deployment.
#
# The Model class will serve multiple users from a its own shared pool of warm GPU containers automatically.
#
# We can deploy this with `modal deploy stable_diffusion_xl.py`.

frontend_path = Path(__file__).parent / "frontend"


@stub.function(
    mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
    allow_concurrent_inputs=20,
)
@asgi_app()
def app():
    import fastapi.staticfiles
    from fastapi import FastAPI

    web_app = FastAPI()

    @web_app.get("/infer/{prompt}")
    async def infer(prompt: str):
        from fastapi.responses import Response

        image_bytes = Model().inference.remote(prompt)

        return Response(image_bytes, media_type="image/png")

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )

    return web_app
