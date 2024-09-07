import sys
sys.path.insert(0, "diffusers/src")

import torch
from diffusers.utils import load_image
from diffusers.pipelines.flux.pipeline_flux_controlnet_inpaint import FluxControlNetInpaintPipeline
from diffusers.models.controlnet_flux import FluxControlNetModel
from diffusers import FluxMultiControlNetModel
from diffusers.utils import load_image, check_min_version
from PIL import Image
import cv2
import numpy as np

from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import requests
import matplotlib.pyplot as plt
import torch.nn as nn
from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector, LeresDetector, DWposeDetector
import os
from pathlib import Path

def get_mask(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]

    mask = (pred_seg>=4) * (pred_seg<=8)
    mask = (255 - mask*255).numpy().astype(np.uint8)
    return mask

canny = CannyDetector()
cloth_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
cloth_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")




base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model = 'YishaoAI/flux-dev-controlnet-canny-kid-clothes'

controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)


pipe = FluxControlNetInpaintPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
#pipe.load_lora_weights("/data/lora.safetensors")
pipe.to("cuda")




image_path = "3.png"
prompt = "children's clothing model"
save_dir = "./save_dir"
os.makedirs(save_dir, exist_ok=True)
mask_path = os.path.join(save_dir, Path(image_path).stem + "_mask.png")

image = load_image(image_path)
mask = get_mask(image, cloth_processor, cloth_model)
cv2.imwrite(mask_path, cv2.merge([mask, mask, mask]))
mask = load_image(mask_path)
#mask = Image.fromarray(cv2.merge([mask, mask, mask]))
canny_image = canny(image)

for scale in range(9, 10):
    scale = scale * 0.1
    for seed in range(10, 100):
        import random
        prompt = prompt + ", indoor" if random.uniform(0,1)<0.5 else prompt + ", outdoor"
        generator = torch.Generator(device="cpu").manual_seed(seed)
        image_res = pipe(
                prompt,
                image=image,
                control_image=canny_image,
                controlnet_conditioning_scale=0.5,
                mask_image=mask,
                strength=0.95,
                num_inference_steps=50,
                guidance_scale=5,
                generator=generator,
                joint_attention_kwargs={"scale": scale},
                ).images[0]

        save_path = os.path.join(save_dir, Path(image_path).stem + f"_{scale}_res_{seed}.png")
        image_res.resize(image.size).save(save_path)

