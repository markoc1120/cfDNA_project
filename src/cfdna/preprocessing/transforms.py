import torch

EPS = 1e-8

TRANSFORM_REGISTRY = {}


def register_transform(name):
    def decorator(fn):
        TRANSFORM_REGISTRY[name] = fn
        return fn

    return decorator


@register_transform('sum_normalization')
def sum_normalization(x: torch.Tensor, **kwargs) -> torch.Tensor:
    return x / (x.sum() + EPS)


@register_transform('log_transform')
def log_transform(x: torch.Tensor, **kwargs) -> torch.Tensor:
    return torch.log(x + EPS)


@register_transform('standardization')
def standardization(
    x: torch.Tensor, train_mean: float = 0.0, train_std: float = 1.0, **kwargs
) -> torch.Tensor:
    return (x - train_mean) / (train_std + EPS)


@register_transform('slice')
def slice_transform(
    x: torch.Tensor,
    ymin: int = 0,
    ymax: int = -1,
    xmin: int = 0,
    xmax: int = -1,
    **kwargs,
) -> torch.Tensor:
    return x[ymin:ymax, xmin:xmax]


def build_transform_pipeline(transform_configs: list[dict]):
    transforms = []
    for tc in transform_configs:
        name = tc['name']
        params = tc.get('params', {})
        fn = TRANSFORM_REGISTRY[name]
        transforms.append((fn, params))

    def pipeline(x: torch.Tensor, **kwargs) -> torch.Tensor:
        for fn, params in transforms:
            merged = {**params, **kwargs}
            x = fn(x, **merged)
        return x

    return pipeline
