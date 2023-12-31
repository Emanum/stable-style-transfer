# from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline as StableDiffusionPipeline
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, \
    StableDiffusionControlNetImg2ImgPipeline
import torch
from diffusers.utils import load_image
from PIL import Image
import cv2
import numpy as np
from transformers import pipeline


# def download_models():
#     ignore = ["*.bin", "*.onnx_data", "*/diffusion_pytorch_model.safetensors"]
#
#     snapshot_download(
#         "runwayml/stable-diffusion-v1-5", ignore_patterns=ignore
#     )


def control_net_canny(img):
    canny_image = canny(img=img, low_threshold=100, high_threshold=200)

    controlnet = get_control_net_canny()
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    image = pipe(
        "the mona lisa", image=canny_image
    ).images[0]
    image.save("mona_lisa.png")


def control_net_depth(img):
    depth_map = extract_depth(img)

    controlnet = get_control_net_depth()

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    image = pipe(
        "lego batman and robin", image=img, control_image=depth_map,
    ).images[0]
    image.save("lego_batman_and_robin.png")


def control_net_combined(img, model_name="runwayml/stable-diffusion-v1-5", prompt="", scale=1.0):
    img = img.resize((int(img.width * scale), int(img.height * scale)))
    depth_map = extract_depth(img)
    canny_image = canny(img=img, low_threshold=100, high_threshold=200)
    # transform to PIL image

    controlnet_canny = get_control_net_canny()
    controlnet_depth = get_control_net_depth()


    pipe = StableDiffusionControlNetImg2ImgPipeline.from_single_file(
        model_name, controlnet=[controlnet_depth, controlnet_canny], torch_dtype=torch.float16,
        use_safetensors=True).to("cuda")

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    image = pipe(
        prompt, image=[img], control_image=[depth_map, canny_image],
    ).images[0]
    image.save(model_name + "_output.png")


def get_control_net_depth():
    return ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16,
                                           use_safetensors=True)


def get_control_net_canny():
    return ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16,
                                           use_safetensors=True)


def canny(img, low_threshold=100, high_threshold=200):
    image = np.array(img)

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)


def get_depth_map(image, depth_estimator):
    image = depth_estimator(image)["depth"]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = torch.from_numpy(image).float() / 255.0
    depth_map = detected_map.permute(2, 0, 1)
    return depth_map


def extract_depth(img):
    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")  # dpt-hybrid-midas
    og_depth_map = get_depth_map(img, depth_estimator).unsqueeze(0).half().to("cuda")
    # save depth map
    depth_map = og_depth_map.squeeze(0).permute(1, 2, 0).cpu().numpy()
    depth_map = depth_map * 255
    depth_map = depth_map.astype(np.uint8)
    depth_map = Image.fromarray(depth_map)
    depth_map.save("depth_map.png")
    return og_depth_map


def test_stable_diffusion():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipe(prompt).images[0]

    image.save("astronaut_rides_horse.png")


# image = load_image("../testData/input_image_vermeer.png")
# canny(image)
# control_net_canny("../testData/input_image_vermeer.png")
#image = load_image("../testData/American_Gothic.jpg")
#control_net_combined(image, model_name="segmind/SSD-1B")

# ReturnOfTheMayflower
# image = load_image("../testData/ReturnOfTheMayflower.jpg")
# control_net_combined(image, model_name="aniflatmixAnimeFlatColorStyle_v20.safetensors", prompt="anime cyberpunk space laser gun")

image = load_image("../testData/TheHuntersInTheSnow.jpg")
control_net_combined(image, model_name="photon_v1.safetensors", prompt="post apocalyptic nuclear winter", scale=0.25)
# control_net_depth(image)
