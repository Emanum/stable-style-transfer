from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline as StableDiffusionPipeline
import torch


# def download_models():
#     ignore = ["*.bin", "*.onnx_data", "*/diffusion_pytorch_model.safetensors"]
#
#     snapshot_download(
#         "runwayml/stable-diffusion-v1-5", ignore_patterns=ignore
#     )


def main():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]

    image.save("astronaut_rides_horse.png")


main()
