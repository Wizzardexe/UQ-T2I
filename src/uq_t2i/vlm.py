from transformers import AutoModelForImageTextToText, AutoProcessor


class QwenVLM:
    def __init__(self, model_id="Qwen/Qwen3-VL-4B-Instruct", device=None):
        self.device = device
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

    def generate(self, image_path, text_prompt, max_new_tokens=256):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text_prompt},
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

        answer = self.processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0]
        return answer
