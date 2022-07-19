#!/usr/bin/env python3

import argparse
import os

import torch
import torch.nn.functional as F

import bitsandbytes as bnb
import PIL.Image
from accelerate import Accelerator
from datasets import load_dataset
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.pipelines import GlidePipeline
import logging
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm
import pandas as pd


class LldDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        return (
            row["images"],
            row["text_embeddings"],
        )


def embed(inputs, tokenizer):
    text_inputs = tokenizer(
        inputs, padding="max_length", max_length=77, return_tensors="pt"
    )
    text_inputs = text_inputs.input_ids.to(accelerator.device)
    with torch.no_grad():
        text_embeddings = accelerator.unwrap_model(text_encoder)(
            text_inputs
        ).last_hidden_state

    return text_embeddings


def augment(image, resolution: int = 64):
    augmentations = Compose(
        [
            Resize(resolution, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(resolution),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.5], [0.5]),
        ]
    )

    return augmentations(image)


def get_dataset(args, tokenizer):
    augmentations = get_augmentations(args)
    df = pd.read_hdf(args.dataset)
    del df["name"]

    df = df.explode("images")

    df["images"] = (
        df["images"]
        .map(lambda img: img.convert("RGB"))
        .map(lambda img: augment(img, args.resolution))
    )

    df["text_embeddings"] = embed(df["description"], tokenizer)

    return df


def main(args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    pipeline = GlidePipeline.from_pretrained("fusing/glide-base")
    model = pipeline.text_unet
    noise_scheduler = DDPMScheduler(timesteps=1000, tensor_format="pt")
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr)

    text_encoder = pipeline.text_encoder.eval()

    dataset = get_dataset(args, pipeline.tokenizer)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs)
        // args.gradient_accumulation_steps,
    )

    (
        model,
        text_encoder,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # Train!
    is_distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    world_size = torch.distributed.get_world_size() if is_distributed else 1
    total_train_batch_size = (
        args.batch_size * args.gradient_accumulation_steps * world_size
    )
    max_steps = (
        len(train_dataloader) // args.gradient_accumulation_steps * args.num_epochs
    )
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataloader.dataset)}")
    print(f"  Num Epochs = {args.num_epochs}")
    print(f"  Instantaneous batch size per device = {args.batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {max_steps}")

    for epoch in range(args.num_epochs):
        model.train()
        with tqdm(total=len(train_dataloader), unit="ba") as pbar:
            pbar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(train_dataloader):
                clean_images = batch["images"]
                batch_size, n_channels, height, width = clean_images.shape
                noise_samples = torch.randn(clean_images.shape).to(clean_images.device)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.timesteps,
                    (batch_size,),
                    device=clean_images.device,
                ).long()

                # add noise onto the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = noise_scheduler.training_step(
                    clean_images, noise_samples, timesteps
                )

                if step % args.gradient_accumulation_steps != 0:
                    with accelerator.no_sync(model):
                        model_output = model(
                            noisy_images, timesteps, batch["text_embeddings"]
                        )
                        model_output, model_var_values = torch.split(
                            model_output, n_channels, dim=1
                        )
                        # Learn the variance using the variational bound, but don't let
                        # it affect our mean prediction.
                        frozen_out = torch.cat(
                            [model_output.detach(), model_var_values], dim=1
                        )

                        # predict the noise residual
                        loss = F.mse_loss(model_output, noise_samples)

                        loss = loss / args.gradient_accumulation_steps

                        accelerator.backward(loss)
                        optimizer.step()
                else:
                    model_output = model(
                        noisy_images, timesteps, batch["text_embeddings"]
                    )
                    model_output, model_var_values = torch.split(
                        model_output, n_channels, dim=1
                    )
                    # Learn the variance using the variational bound, but don't let
                    # it affect our mean prediction.
                    frozen_out = torch.cat(
                        [model_output.detach(), model_var_values], dim=1
                    )

                    # predict the noise residual
                    loss = F.mse_loss(model_output, noise_samples)
                    loss = loss / args.gradient_accumulation_steps
                    accelerator.backward(loss)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                pbar.update(1)
                pbar.set_postfix(
                    loss=loss.detach().item(), lr=optimizer.param_groups[0]["lr"]
                )

        accelerator.wait_for_everyone()

        # Generate a sample image for visual inspection
        if accelerator.is_main_process:
            model.eval()
            with torch.no_grad():
                pipeline.unet = accelerator.unwrap_model(model)

                generator = torch.manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                image = pipeline(
                    "a clip art of a corgi",
                    generator=generator,
                    num_upscale_inference_steps=50,
                )

            # process image to PIL
            image_processed = image.squeeze(0)
            image_processed = (
                ((image_processed + 1) * 127.5)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            image_pil = PIL.Image.fromarray(image_processed)

            # save image
            test_dir = os.path.join(args.output_dir, "test_samples")
            os.makedirs(test_dir, exist_ok=True)
            image_pil.save(f"{test_dir}/{epoch:04d}.png")

            pipeline.save_pretrained(args.output_dir)
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dataset", type=str, default="./data/lld-processed.h5")
    parser.add_argument("--output_dir", type=str, default="glide-text2image")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", action="store_true")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    main(args)
