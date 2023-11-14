from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline as StableDiffusionPipeline
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

def canny(imgfile, low_threshold=100, high_threshold=200):
    image = load_image(imgfile)

    image = np.array(image)

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def main():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]

    image.save("astronaut_rides_horse.png")


canny("../testData/input_image_vermeer.png")
