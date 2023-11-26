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
        "lllyasviel/sd-controlnet-canny", ignore_patterns=ignore
    )
    snapshot_download(
        "lllyasviel/control_v11f1p_sd15_depth", ignore_patterns=ignore
    )
    snapshot_download(
        "Intel/dpt-hybrid-midas", ignore_patterns=ignore
    )

    # custom models from civitai.com
    run_async(download_file_with_tqdm, "https://civitai.com/api/download/models/1356",
              "dreamlikeDiffusion10_10.ckpt")
    run_async(download_file_with_tqdm, "https://civitai.com/api/download/models/90072",
              "photon_v1.safetensors")
    run_async(download_file_with_tqdm, "https://civitai.com/api/download/models/40816",
              "aniflatmixAnimeFlatColorStyle_v20.safetensors")
    run_async(download_file_with_tqdm, "https://civitai.com/api/download/models/105924",
              "cetusMix_Whalefall2.safetensors")
    run_async(download_file_with_tqdm,
              "https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor&size=pruned&fp=fp16",
              "realisticVisionV51_v51VAE.safetensors")
    run_async(download_file_with_tqdm,
              "https://civitai.com/api/download/models/198962?type=Model&format=SafeTensor&size=pruned&fp=fp16",
              "dynavisionXLAllInOneStylized_release0557Bakedvae.safetensors")


def run_async(func, *args, **kwargs):
    import threading
    thread = threading.Thread(target=func, args=args, kwargs=kwargs)
    thread.start()
    return thread


def download_file_with_tqdm(url, filename):
    import requests
    from pathlib import Path
    from tqdm import tqdm

    r = requests.get(url, allow_redirects=True, stream=True)
    with open(filename, 'wb') as f:
        file_size = int(r.headers['Content-Length'])
        chunk_size = 1024
        num_bars = int(file_size / chunk_size)
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=num_bars, unit='KB', desc=filename, leave=True):
            f.write(chunk)


image = (
    Image.debian_slim()
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "diffusers",
        # "invisible_watermark",
        "transformers",
        "accelerate",
        "safetensors",
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
    from diffusers.utils import load_image
    import torch
    from PIL import Image as PILImage
    from diffusers import StableDiffusionControlNetImg2ImgPipeline, UniPCMultistepScheduler, \
        StableDiffusionXLControlNetImg2ImgPipeline
    import pathlib


# ## Load model and run inference
#
# The container lifecycle [`__enter__` function](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta)
# loads the model at startup. Then, we evaluate it in the `run_inference` function.
#
# To avoid excessive cold-starts, we set the idle timeout to 240 seconds, meaning once a GPU has loaded the model it will stay
# online for 4 minutes before spinning down. This can be adjusted for cost/experience trade-offs.


@stub.cls(gpu=gpu.A10G(), container_idle_timeout=240)
class Model:
    def __init__(self):
        self.pipe = None

    def __enter__(self):
        import torch
        from diffusers import DiffusionPipeline, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, \
            UniPCMultistepScheduler, StableDiffusionPipeline
        from transformers import pipeline

        load_options = dict(
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
        )

        # Load base model

        # Load ControlNet
        print("Loading ControlNet Canny")
        self.canny_controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",
                                                                torch_dtype=torch.float16,
                                                                use_safetensors=True)

        print("Loading ControlNet Depth")
        self.depth_controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth",
                                                                torch_dtype=torch.float16,
                                                                use_safetensors=True)

        # print("Loading ControlNetXL Depth")
        # self.depth_controlnetXL = ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0",
        #                                                         variant="fp16",
        #                                                         use_safetensors=True,
        #                                                         torch_dtype=torch.float16,
        #                                                         ).to("cuda")
        print("Loading ControlNet depth-estimation")
        self.depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")

        # Compiling the model graph is JIT so this will increase inference time for the first run
        # but speed up subsequent runs. Uncomment to enable.
        # self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        # self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)

    @method()
    def inference(self, prompt: str, init_image_bytes: bytes, guess_mode: bool = False, canny: bool = False,
                  model_name: str = "aniflatmixAnimeFlatColorStyle_v20.safetensors", creativity: int = 50):
        # negative_prompt = "disfigured, ugly, deformed"

        # write image to file
        with open("image.png", "wb") as f:
            f.write(init_image_bytes)

        init_image = load_image("image.png")
        # reduce image size to max 1024px on the longest side but keep aspect ratio
        max_size = 1024
        if init_image.width > max_size or init_image.height > max_size:
            print(f"Resizing image from {init_image.width}x{init_image.height}")
            if init_image.width > init_image.height:
                new_width = max_size
                new_height = int(max_size * init_image.height / init_image.width)
            else:
                new_height = max_size
                new_width = int(max_size * init_image.width / init_image.height)
            init_image = init_image.resize((new_width, new_height))
            print(f"Resized image to {new_width}x{new_height}")

        # calc canny
        print("Calculating canny")
        canny_img = np.array(init_image)
        canny_img = cv2.Canny(canny_img, 100, 200)
        canny_img = canny_img[:, :, None]
        canny_img = np.concatenate([canny_img, canny_img, canny_img], axis=2)
        canny_img = PILImage.fromarray(canny_img)
        # canny_pil_img = PILImage.fromarray(canny_img[0])

        # calc depth_map
        # depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")  # dpt-hybrid-midas
        print("Calculating depth map")
        og_depth_map = Model().get_depth_map.remote(init_image, self.depth_estimator)
        depth_map = og_depth_map.unsqueeze(0).half().to("cuda")
        # depth_img = og_depth_map.squeeze(0).permute(1, 2, 0).cpu().numpy()
        # depth_img = depth_img * 255
        # depth_img = depth_img.astype(np.uint8)
        # depth_img = PILImage.fromarray(depth_img)

        # load model
        print("Loading Pipeline")
        # check if XL in model name
        if "XL" in model_name:
            self.pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                pathlib.Path("./" + model_name),
                # controlnet=[self.depth_controlnet, self.canny_controlnet],
                controlnet=[self.depth_controlnetXL],
                torch_dtype=torch.float16,
                use_safetensors=(model_name.endswith(".safetensors")),
                safety_checker=None,
                load_safety_checker=False).to("cuda")
        else:
            control_net = [self.depth_controlnet]
            if canny:
                control_net.append(self.canny_controlnet)
            self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
                model_name,
                controlnet=control_net,
                torch_dtype=torch.float16,
                use_safetensors=(model_name.endswith(".safetensors"))
            ).to("cuda")

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()

        # convert creativity to guidance_scale, inverse proportional
        # min_guidance_scale = 5, max_guidance_scale = 15
        # cast creativity to float
        min_guidance_scale = 1
        max_guidance_scale = 25

        creativity = float(creativity)
        guidance_scale = min_guidance_scale + (max_guidance_scale - min_guidance_scale) * (1 - creativity / 100)
        print(f"Guidance scale: {guidance_scale}")

        print("Running Pipeline")
        control_img = [depth_map]
        if canny:
            control_img.append(canny_img)
        image = self.pipe(
            prompt,
            image=init_image,
            control_image=control_img,
            guess_mode=guess_mode,
            guidance_scale=guidance_scale,
        )
        image = image.images[0]
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
def main(prompt: str, init_image: bytes, guess_mode: bool = False,
         model_name: str = "aniflatmixAnimeFlatColorStyle_v20.safetensors"):
    image_bytes = Model().inference.remote(prompt, init_image, guess_mode, model_name)

    dir = Path("output")
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

    @web_app.post("/infer/")
    async def infer(request: fastapi.Request):
        from fastapi.responses import Response

        form = await request.form()
        init_image = await form["init_image"].read()
        prompt = form["prompt"]
        guess_mode = form["guess_mode"]
        model_name = form["model_name"]
        creativity = form["creativity"]
        canny = form["canny"]

        image_bytes = Model().inference.remote(prompt, init_image, guess_mode, canny, model_name, creativity)

        return Response(image_bytes, media_type="image/png")

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )

    return web_app
