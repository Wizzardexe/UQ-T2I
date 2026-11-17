import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro

from uq_t2i.t2i_metrics import VQAScore
from uq_t2i.utils import (
    add_metrics,
    caption_without_punc,
    connect,
    images_without_caption,
    images_without_vqascore,
    prompts_without_images,
    register_prompts,
)

from .generate import SD1_5
from .punc import punc_scores
from .vlm import QwenVLM

SEEDS = [0, 1, 2, 42, 1234, 2023, 9999]


@dataclass
class Args:
    all_prompts: bool = False
    """Use all prompts instead of a small example set."""
    generate_images: bool = True
    """Generate images for the prompts."""
    image_folder: Path = "examples"
    """The folder containing the generated images."""
    MODEL_ID: str = "sd-legacy/stable-diffusion-v1-5"
    """The model ID for the T2I model."""
    STEPS: int = 50
    """The number of inference steps for the T2I model."""
    GUIDANCE: float = 7.5
    """The guidance scale for the T2I model."""
    VQA: bool = True
    """Whether to run VQAScore on the generated images."""
    MODEL_ID_VQA: str = "qwen3.5-4b"
    """The model ID for the VQA model."""
    MODEL_VLM: str = "Qwen/Qwen3.5-4B"
    """The model ID for the VLM model."""
    generate_captions: bool = True
    """Whether to generate captions for the images."""
    instruction: str = "How would you describe the contents of this photo?"
    """The instruction to use for the caption generation model."""

    PUNC: bool = True
    """Whether to run PUNC on the generated images."""


def device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def run(args: Any) -> None:
    device, dtype = device_and_dtype()

    OUT_IMG = os.path.join(os.getcwd(), "outputs", args.image_folder)
    OUT_DB = os.path.join(os.getcwd(), "outputs", f"{args.image_folder}.duckdb")
    with connect(OUT_DB) as con:
        register_prompts(con)

    count_files = len(os.listdir(OUT_IMG)) if os.path.exists(OUT_IMG) else 0
    if args.generate_images:
        stable_diffusion = SD1_5(
            name=args.MODEL_ID,
            device=device,
            steps=args.STEPS,
            guidance=args.GUIDANCE,
            dtype=dtype,
            output_dir=OUT_IMG,
            db_path=OUT_DB,
            run=args.image_folder,
        )
        with connect(OUT_DB) as con:
            prompts = prompts_without_images(con, run=args.image_folder)
        stable_diffusion.generate_batch(prompts, SEEDS)
    elif not args.generate_images and count_files > 0:
        print("Skipping image generation.")
        images = [f for f in os.listdir(OUT_IMG) if f.endswith(".png")]
        print(f"Found {len(images)} images in {OUT_IMG}.")
    else:
        raise FileNotFoundError(
            f"Image folder {args.image_folder} does not exist or is empty."
        )
    if args.generate_captions:
        qwen_vlm = QwenVLM(model_id=args.MODEL_VLM, OUT_DB=OUT_DB)
        with connect(OUT_DB) as con:
            images = images_without_caption(con, args.MODEL_VLM, args.instruction)
        qwen_vlm.generate(images, args.instruction)
    else:
        print("Skipping caption generation.")

    # punc scores
    if args.punc:
        with connect(OUT_DB) as con:
            df_captions = caption_without_punc(con, args.MODEL_VLM)
        punc_scores(df_captions, OUT_DB)

    # VQA scores
    if args.vqascore:
        qwen_score = VQAScore(model=args.MODEL_ID_VQA, device=device)
        with connect(OUT_DB) as con:
            images_without_vqa = images_without_vqascore(con)
        images = images_without_vqa["path"].tolist()
        prompts = images_without_vqa["text"].tolist()
        print(f"Found {len(images)} images without VQA scores.")
        scores = qwen_score.score(images, prompts)
        for image_id, score in zip(images_without_vqa["image_id"], scores):
            with connect(OUT_DB) as con:
                add_metrics(
                    con,
                    [
                        {
                            "image_id": image_id,
                            "name": "vqa",
                            "metric": "score",
                            "value": score,
                        }
                    ],
                )


if __name__ == "__main__":
    cli_args = tyro.cli(Args)
    run(cli_args)
