import torch
import sys
sys.path.insert(0, "diffusers/src")

from diffusers import DiffusionPipeline
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to(device)

prompt = "children's clothing model, boy"
generator = torch.Generator(device="cpu").manual_seed(42)
#image = pipe(prompt, generator=generator, guidance_scale=3.5).images[0]
#image.save("no_lora.png")


pipe.load_lora_weights("/data/lora.safetensors")
generator = torch.Generator(device="cpu").manual_seed(42)

for seed in range(2000, 2100):
    prompt = prompt + ", indoor" if random.uniform(0,1)<0.5 else prompt + ", outdoor"
    scale = 0.8
    generator = torch.Generator("cuda").manual_seed(seed)
    image = pipe(prompt, generator=generator, joint_attention_kwargs={"scale": scale}, guidance_scale=3.5, num_inference_steps=50).images[0]
    image.save(f"lora_1_{scale}_{seed}.png")
