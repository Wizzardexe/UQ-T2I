import os
import re

import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import make_image_grid

from .utils import add_images, connect, image_id


class SD1_5:
    def __init__(
        self,
        name="sd-legacy/stable-diffusion-v1-5",
        device=None,
        steps=50,
        guidance=7.5,
        dtype=torch.float16,
        output_dir="outputs",
        db_path="outputs/experiment.duckdb",
        run="main",
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
        self.output_dir = output_dir
        self.db_path = db_path
        self.run = run

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

    def generate_batch(self, prompts, SEEDS, flush_every=50):
        OUT_IMG = self.output_dir
        os.makedirs(OUT_IMG, exist_ok=True)
        buffer = []
        for i, (pid, prompt, group) in enumerate(prompts.itertuples(index=False)):
            images = []

            for seed in SEEDS:
                image = self.generate(prompt, seed, self.steps, self.guidance)
                name = f"{i:03d}_{group}_{self.slug(prompt)}_seed{seed}.png"
                path = os.path.join(OUT_IMG, name)
                (image.save(path),)
                images.append(image)
                buffer.append(
                    {
                        "image_id": image_id(pid, seed),
                        "prompt_id": pid,
                        "seed": seed,
                        "run": self.run,
                        "model": "sd-legacy/stable-diffusion-v1-5",
                        "steps": self.steps,
                        "guidance": self.guidance,
                        "path": path,
                    }
                )
            grid = make_image_grid(images, rows=1, cols=len(images))
            grid.save(
                os.path.join(OUT_IMG, f"{i:03d}_{group}_{self.slug(prompt)}_grid.png")
            )
            print(f"[{i + 1}/{len(prompts)}] {group}: {prompt!r}")
            if len(buffer) >= flush_every:
                self._write(buffer)
                buffer = []
        self._write(buffer)

    def _write(self, rows):
        if rows:
            with connect(self.out_db) as con:
                add_images(con, rows)
