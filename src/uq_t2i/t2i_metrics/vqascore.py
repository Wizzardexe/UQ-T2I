import os

from .models.vqascore_models import get_vqascore_model, list_all_vqascore_models
from .score import Score


class VQAScore(Score):
    def prepare_scoremodel(
        self, model="clip-flant5-xxl", device="cuda", cache_dir=None, **kwargs
    ):
        self.cache_dir = cache_dir or os.environ.get("HF_HUB_CACHE")

        return get_vqascore_model(
            model, device=device, cache_dir=self.cache_dir, **kwargs
        )

    def list_all_models(self) -> list[str]:
        return list_all_vqascore_models()
