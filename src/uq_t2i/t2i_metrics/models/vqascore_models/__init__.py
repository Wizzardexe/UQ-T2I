from ...constants import HF_CACHE_DIR
from .qwen3vl_model import QWEN3_VL_MODELS, Qwen3VLModel

ALL_VQA_MODELS = [QWEN3_VL_MODELS]


def list_all_vqascore_models():
    return [model for models in ALL_VQA_MODELS for model in models]


def get_vqascore_model(model_name, device="cuda", cache_dir=HF_CACHE_DIR, **kwargs):
    assert model_name in list_all_vqascore_models(), f"Unknown model: {model_name}"
    return Qwen3VLModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
