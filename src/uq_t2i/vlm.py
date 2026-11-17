import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .utils import add_captions, connect


class QwenVLM:
    PUNC_INSTRUCTION = "How would you describe the contents of this photo?"

    def __init__(self, model_id="Qwen/Qwen3.5-4B", cache_dir=None, out_db=None):
        cache_dir = cache_dir or os.environ.get("HF_HUB_CACHE")
        self.model_id = model_id
        self.out_db = out_db
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype="auto", device_map="auto", cache_dir=cache_dir
        )
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

    @torch.inference_mode()
    def caption_one(self, image_path, instruction, max_new_tokens=256):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0].strip()

    def generate(
        self, images, instruction=None, max_new_tokens=256, punc=False, flush_every=50
    ):
        if punc:
            instruction = self.PUNC_INSTRUCTION
        if not instruction:
            raise ValueError("instruction is required unless punc=True")

        buffer = []
        for image_id, image_path in images[["image_id", "path"]].itertuples(
            index=False
        ):
            buffer.append(
                {
                    "image_id": image_id,
                    "captioner": self.model_id,
                    "instruction": instruction,
                    "caption": self.caption_one(
                        image_path, instruction, max_new_tokens
                    ),
                }
            )
            if len(buffer) >= flush_every:
                self._write(buffer)
                buffer = []
        self._write(buffer)

    def _write(self, rows):
        if rows:
            with connect(self.out_db) as con:
                add_captions(con, rows)
