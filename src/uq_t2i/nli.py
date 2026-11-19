import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .utils import add_metrics, connect


class NLI:
    def __init__(
        self,
        name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        cache_dir: str | None = None,
        out_db: str | None = None,
    ):
        self.name = name
        self.cache_dir = cache_dir or os.environ.get("HF_HUB_CACHE")
        self.tok = AutoTokenizer.from_pretrained(name, cache_dir=self.cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, cache_dir=self.cache_dir
        ).eval()
        self.device = device
        self.model.to(self.device)
        self.out_db = out_db

    def nli(self, df_captions, batch_size=32):
        prompts = df_captions["prompt"].tolist()
        captions = df_captions["caption"].tolist()
        image_ids = df_captions["image_id"].tolist()
        labels = [
            self.model.config.id2label[k].lower()
            for k in range(self.model.config.num_labels)
        ]

        for start in range(0, len(prompts), batch_size):
            end = start + batch_size
            x = self.tok(
                prompts[start:end],
                captions[start:end],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            with torch.inference_mode():
                probs = self.model(**x).logits.softmax(-1).cpu().tolist()

            rows = [
                {
                    "image_id": img_id,
                    "name": "nli",
                    "metric": label,
                    "value": round(prob, 3),
                }
                for img_id, row in zip(image_ids[start:end], probs)
                for label, prob in zip(labels, row)
            ]
            with connect(self.out_db) as con:
                add_metrics(con, rows)
