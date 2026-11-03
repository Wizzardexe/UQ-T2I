import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro
from diffusers import StableDiffusionPipeline
from diffusers.utils import make_image_grid

from .prompts import example_prompts, generate_prompts_list

MODEL_ID = "sd-legacy/stable-diffusion-v1-5"
SEEDS = [0, 1, 2, 42, 1234, 2023, 9999]
STEPS = 50
GUIDANCE = 7.5


@dataclass
class Args:
    all_prompts: bool = False
    """Use all prompts instead of a small example set."""
    generate: bool = True
    """Generate images for the prompts."""
    image_folder: Path = "examples"
    """The folder containing the generated images."""


def device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def slug(text, max_len=60):
    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return s[:max_len] or "empty_prompt"


def text_to_image(
    prompt: str, seed: int, pipe: StableDiffusionPipeline
) -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(seed)
    image = pipe(
        prompt,
        generator=generator,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
    ).images[0]
    return image


def run(args: Any) -> None:
    device, dtype = device_and_dtype()
    if args.generate:
        OUT = os.path.join(os.getcwd(), args.image_folder)
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to(
            device
        )
        if device == "mps":
            pipe.enable_attention_slicing()
        pipe.set_progress_bar_config(disable=True)

        prompts = generate_prompts_list() if args.all_prompts else example_prompts()
        print(
            f"Generating {len(prompts)} prompts with {len(SEEDS)} seeds each ({len(prompts) * len(SEEDS)} images total)."
        )

        os.makedirs(OUT, exist_ok=True)

        with open(OUT / "manifest.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "group", "prompt", "seed"])
            for i, (group, prompt) in enumerate(prompts):
                images = []
                for seed in SEEDS:
                    image = text_to_image(prompt, seed, pipe)
                    name = f"{i:03d}_{group}_{slug(prompt)}_seed{seed}.png"
                    image.save(OUT / name)
                    writer.writerow([name, group, prompt, seed])
                    images.append(image)
                grid = make_image_grid(images, rows=1, cols=len(images))
                grid.save(OUT / f"{i:03d}_{group}_{slug(prompt)}_grid.png")
                print(f"[{i + 1}/{len(prompts)}] {group}: {prompt!r}")
    else:
        print("Skipping image generation.")
        images = []


if __name__ == "__main__":
    cli_args = tyro.cli(Args)
    run(cli_args)
