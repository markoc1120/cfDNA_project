import numpy
from .cnn_model import RebinnedCNNModel
from .mlp_model import MLPMultipleInputsModel

MODEL_REGISTRY = {
    'cnn_model': RebinnedCNNModel,
    'mlp_model': MLPMultipleInputsModel,
}

def get_model(model_type: str, n_inputs=None):
    if model_type == 'mlp_model':
        if n_inputs is None:
            raise Exception('n_inputs is None, while model_type is mlp_model')
        return MODEL_REGISTRY[model_type](n_inputs=n_inputs)
    else:
        return MODEL_REGISTRY[model_type]()
