#!/usr/bin/env python3

import torch
from diffusers import UNetUnconditionalModel, DDIMScheduler
import PIL.Image
import numpy as np
import tqdm

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load models
scheduler = DDIMScheduler.from_config("fusing/ddpm-celeba-hq", tensor_format="pt")
unet = UNetUnconditionalModel.from_pretrained("fusing/ddpm-celeba-hq", ddpm=True).to(
    torch_device
)

# 2. Sample gaussian noise
generator = torch.manual_seed(23)
unet.image_size = unet.resolution
image = torch.randn(
    (1, unet.in_channels, unet.image_size, unet.image_size),
    generator=generator,
)
image = image.to(torch_device)

# 3. Denoise
num_inference_steps = 50
eta = 0.0  # <- deterministic sampling
scheduler.set_timesteps(num_inference_steps)

for t in tqdm.tqdm(scheduler.timesteps):
    # 1. predict noise residual
    with torch.no_grad():
        residual = unet(image, t)["sample"]

    prev_image = scheduler.step(residual, t, image, eta)["prev_sample"]

    # 3. set current image to prev_image: x_t -> x_t-1
    image = prev_image

# 4. process image to PIL
image_processed = image.cpu().permute(0, 2, 3, 1)
image_processed = (image_processed + 1.0) * 127.5
image_processed = image_processed.numpy().astype(np.uint8)
image_pil = PIL.Image.fromarray(image_processed[0])

# 5. save image
image_pil.show()
image_pil.save("generated_image.png")
