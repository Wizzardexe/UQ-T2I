from transformers import AutoModelForImageTextToText, AutoProcessor

from .utils import add_captions, connect


class QwenVLM:
    def __init__(
        self,
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        device=None,
        cache_dir=None,
        OUT_DB=None,
    ):
        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype="auto", device_map="auto", cache_dir=cache_dir
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.cache_dir = cache_dir
        self.punc_instruction = "How would you describe the contents of this photo?"
        self.OUT_DB = OUT_DB
        self.model_id = model_id

    def generate(self, images, text_instruction, max_new_tokens=256, punc=False):
        if punc:
            text_instruction = self.punc_instruction
        for image_id, image_path in images.itertuples(index=False):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": text_instruction},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                cache_dir=self.cache_dir,
            ).to(self.model.device)

            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

            answer = self.processor.batch_decode(
                output_ids[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )[0]
            with connect(self.OUT_DB) as con:
                add_captions(
                    con,
                    [
                        {
                            "image_id": image_id,
                            "captioner": f"self.{self.model_id}",
                            "instruction": text_instruction,
                            "caption": answer,
                        }
                    ],
                )
