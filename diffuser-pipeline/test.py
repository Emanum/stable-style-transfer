import torch
from diffusers import DiffusionPipeline, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, \
    UniPCMultistepScheduler
from transformers import pipeline
import numpy as np
import cv2
from PIL import Image
from diffusers.utils import load_image
import torch


class Model:
    def __int__(self):
        self.canny_controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",
                                                                torch_dtype=torch.float16,
                                                                use_safetensors=True)

        self.depth_controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth",
                                                                torch_dtype=torch.float16,
                                                                use_safetensors=True)

        self.depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=[self.depth_controlnet, self.canny_controlnet],
            torch_dtype=torch.float16,
            use_safetensors=True).to("cuda")

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()

    def transform(self, prompt):
        # negative_prompt = "disfigured, ugly, deformed"

        # load image from url
        init_image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

        # calc canny
        canny_img = np.array(init_image)
        canny_img = cv2.Canny(canny_img, 100, 200)
        canny_img = canny_img[:, :, None]
        canny_img = np.concatenate([canny_img, canny_img, canny_img], axis=2)
        # warp canny in another list
        canny_img = np.array([canny_img])
        # canny_image = Image.fromarray(canny_img)

        # calc depth_map
        # depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")  # dpt-hybrid-midas
        depth_map = self.get_depth_map(init_image, self.depth_estimator).unsqueeze(0).half().to("cuda")
        # depth_map = self.get_depth_map(init_image, self.depth_estimator).unsqueeze(0).half().to("cuda")

        print(f"type of init_image: {type(init_image)}")
        list_of_images = [init_image]
        image = self.pipe(
            prompt, image=list_of_images, control_image=[depth_map, canny_img],
        )
        image = image.images[0]
        # image = canny_image

        import io

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="PNG")
        image_bytes = byte_stream.getvalue()

        return image_bytes

    def get_depth_map(self, image, depth_estimator):
        image = depth_estimator(image)["depth"]
        image = np.array(image)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        detected_map = torch.from_numpy(image).float() / 255.0
        depth_map = detected_map.permute(2, 0, 1)
        return depth_map


if __name__ == "__main__":
    model = Model()
    model.__int__()
    transform = model.transform("cat")
    # save to file
    with open("output.png", "wb") as f:
        f.write(transform)
