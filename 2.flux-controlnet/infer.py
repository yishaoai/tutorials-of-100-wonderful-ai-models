import sys
sys.path.insert(0, "diffusers/src")

import torch
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel
from controlnet_aux import CannyDetector

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model = 'YishaoAI/flux-dev-controlnet-canny-kid-clothes'

controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda")

canny = CannyDetector()

image_path = "3.png"
prompt = "children's clothing model"
save_dir = "./save_dir"
os.makedirs(save_dir, exist_ok=True)

image = load_image(image_path)
canny_image = canny(image)
image = pipe(
    prompt, 
    control_image=canny_image,
    controlnet_conditioning_scale=0.6,
    num_inference_steps=28, 
    guidance_scale=3.5,
).images[0]


image.save("image.jpg")


