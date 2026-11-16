import csv
import os
import re

import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import make_image_grid


class SD1_5:
    def __init__(
        self,
        name="sd-legacy/stable-diffusion-v1-5",
        device=None,
        steps=50,
        guidance=7.5,
        dtype=torch.float16,
    ):
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(name, torch_dtype=dtype).to(
            self.device
        )
        if self.device == "mps":
            self.pipe.enable_attention_slicing()
        self.pipe.set_progress_bar_config(disable=True)
        self.steps = steps
        self.guidance = guidance
        self.dtype = dtype

    def generate(self, prompt, seed, steps=50, guidance=7.5):
        generator = torch.Generator(self.device).manual_seed(seed)
        image = self.pipe(
            prompt,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=guidance,
        ).images[0]
        return image

    @staticmethod
    def slug(text, max_len=60):
        s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
        return s[:max_len] or "empty_prompt"

    def generate_batch(self, prompts, SEEDS, OUT):
        os.makedirs(OUT, exist_ok=True)

        with open(os.path.join(OUT, "manifest.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "group", "prompt", "seed"])
            for i, (group, prompt) in enumerate(prompts):
                images = []
                for seed in SEEDS:
                    image = self.generate(prompt, seed, self.steps, self.guidance)
                    name = f"{i:03d}_{group}_{self.slug(prompt)}_seed{seed}.png"
                    image.save(os.path.join(OUT, name))
                    writer.writerow([name, group, prompt, seed])
                    images.append(image)
                grid = make_image_grid(images, rows=1, cols=len(images))
                grid.save(
                    os.path.join(OUT, f"{i:03d}_{group}_{self.slug(prompt)}_grid.png")
                )
                print(f"[{i + 1}/{len(prompts)}] {group}: {prompt!r}")
