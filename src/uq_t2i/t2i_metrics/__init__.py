from .constants import HF_CACHE_DIR
from .vqascore import VQAScore, list_all_vqascore_models


def list_all_models():
    return list_all_vqascore_models()


def get_score_model(
    model="qwen3.5-4b", device="cuda", cache_dir=HF_CACHE_DIR, **kwargs
):
    if model in list_all_vqascore_models():
        return VQAScore(model, device=device, cache_dir=cache_dir, **kwargs)
    raise NotImplementedError()
