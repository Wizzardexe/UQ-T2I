import csv
import re
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import make_image_grid

from .prompts import ambiguous_prompts, generate_prompts_list, novel_compositions, paraphrases, underspecified_prompts

MODEL_ID = "sd-legacy/stable-diffusion-v1-5"
SEEDS = [0, 1, 2, 42, 1234, 2023, 9999]
STEPS = 50
GUIDANCE = 7.5
OUT = Path("outputs/examples")


def example_prompts():
    amb = "a photo of the bat on a table"
    under = "a dog in the park"
    return ([("ambiguous", amb)]
            + [("clarification", c) for c in ambiguous_prompts[amb]]
            + [("underspecified", under), ("specified", underspecified_prompts[under][0])]
            + [("paraphrase", p) for p in paraphrases[0]]
            + [("composition", p) for p in novel_compositions[:2]])


def device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def slug(text, max_len=60):
    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return s[:max_len] or "empty_prompt"


def main(all_prompts: bool = False) -> None:
    device, dtype = device_and_dtype()
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    if device == "mps":
        pipe.enable_attention_slicing()   
    pipe.set_progress_bar_config(disable=True)

    prompts = generate_prompts_list() if all_prompts else example_prompts()
    print(f"Generating {len(prompts)} prompts with {len(SEEDS)} seeds each ({len(prompts) * len(SEEDS)} images total).")
    OUT.mkdir(parents=True, exist_ok=True)

    with open(OUT / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "group", "prompt", "seed"])
        for i, (group, prompt) in enumerate(prompts):
            images = []
            for seed in SEEDS:
                generator = torch.Generator("cpu").manual_seed(seed)
                image = pipe(prompt, generator=generator, num_inference_steps=STEPS,
                             guidance_scale=GUIDANCE).images[0]
                name = f"{i:03d}_{group}_{slug(prompt)}_seed{seed}.png"
                image.save(OUT / name)
                writer.writerow([name, group, prompt, seed])
                images.append(image)
            grid = make_image_grid(images, rows=1, cols=len(images))
            grid.save(OUT / f"{i:03d}_{group}_{slug(prompt)}_grid.png")
            print(f"[{i + 1}/{len(prompts)}] {group}: {prompt!r}")


if __name__ == "__main__":
    main(all_prompts="--all" in sys.argv)