from .cnn_model import RebinnedCNNModel
from .mlp_model import MLPMultipleInputsModel

MODEL_REGISTRY = {
    'cnn': RebinnedCNNModel,
    'mlp': MLPMultipleInputsModel,
}


def get_model(model_type: str, **kwargs):
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f'Unknown model type: {model_type}. Available: {available}')
    return MODEL_REGISTRY[model_type](**kwargs)
