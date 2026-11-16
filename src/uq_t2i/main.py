import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro

from uq_t2i.t2i_metrics import VQAScore

from .generate import SD1_5
from .prompts import example_prompts, generate_prompts_list

SEEDS = [0, 1, 2, 42, 1234, 2023, 9999]


@dataclass
class Args:
    all_prompts: bool = False
    """Use all prompts instead of a small example set."""
    generate: bool = True
    """Generate images for the prompts."""
    image_folder: Path = "examples"
    """The folder containing the generated images."""
    MODEL_ID: str = "sd-legacy/stable-diffusion-v1-5"
    """The model ID for the T2I model."""
    STEPS: int = 50
    """The number of inference steps for the T2I model."""
    GUIDANCE: float = 7.5
    """The guidance scale for the T2I model."""
    MODEL_ID_VQA: str = "qwen3.5-4b"
    """The model ID for the VQA model."""


def device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def run(args: Any) -> None:
    device, dtype = device_and_dtype()
    OUT = os.path.join(os.getcwd(), "outputs", args.image_folder)
    count_files = len(os.listdir(OUT)) if os.path.exists(OUT) else 0
    if args.generate:
        stable_diffusion = SD1_5(
            name=args.MODEL_ID,
            device=device,
            steps=args.STEPS,
            guidance=args.GUIDANCE,
            dtype=dtype,
        )

        prompts = generate_prompts_list() if args.all_prompts else example_prompts()
        print(
            f"Generating {len(prompts)} prompts with {len(SEEDS)} seeds each ({len(prompts) * len(SEEDS)} images total)."
        )

        stable_diffusion.generate_batch(prompts, SEEDS, OUT)
    elif not args.generate and count_files > 0:
        print("Skipping image generation.")
        images = [f for f in os.listdir(OUT) if f.endswith(".png")]
        print(f"Found {len(images)} images in {OUT}.")
    else:
        raise FileNotFoundError(
            f"Image folder {args.image_folder} does not exist or is empty."
        )

    qwen_score = VQAScore(
        model=args.MODEL_ID_VQA, device=device, cache_dir=os.environ["VQA_CACHE"]
    )
    images = [
        os.path.join(
            OUT,
            "000_ambiguous_a_girl_looking_at_a_boy_with_a_telescope_seed0.png",
        )
    ]
    texts = ["a girl looking at a boy with a telescope"]
    scores = qwen_score(images=images, texts=texts)
    print(f"VQA scores: {scores}")


if __name__ == "__main__":
    cli_args = tyro.cli(Args)
    run(cli_args)
