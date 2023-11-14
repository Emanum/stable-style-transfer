from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline as StableDiffusionPipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from diffusers.utils import load_image
from PIL import Image
import cv2
import numpy as np


# def download_models():
#     ignore = ["*.bin", "*.onnx_data", "*/diffusion_pytorch_model.safetensors"]
#
#     snapshot_download(
#         "runwayml/stable-diffusion-v1-5", ignore_patterns=ignore
#     )

def canny(img, low_threshold=100, high_threshold=200):
    image = load_image(img)
    image = np.array(image)

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def control_net(img):
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16,
                                                 use_safetensors=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    canny_image = canny(img=img, low_threshold=100, high_threshold=200)

    image = pipe(
        "the mona lisa", image=canny_image
    ).images[0]
    image.save("mona_lisa.png")


def main():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]

    image.save("astronaut_rides_horse.png")


# canny("../testData/input_image_vermeer.png")
control_net("../testData/input_image_vermeer.png")
