import torch

from cfdna.preprocessing.transforms import (
    TRANSFORM_REGISTRY,
    build_transform_pipeline,
    log_transform,
    slice_transform,
    standardization,
    sum_normalization,
)

SAMPLE_2X2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])


def test_registry_has_all_transforms():
    for name in ('sum_normalization', 'log_transform', 'standardization', 'slice'):
        assert name in TRANSFORM_REGISTRY


def test_sum_normalization():
    result = sum_normalization(SAMPLE_2X2)
    assert abs(result.sum().item() - 1.0) < 1e-5


def test_sum_normalization_zeros():
    assert sum_normalization(torch.zeros(3, 3)).sum().item() < 1e-5


def test_log_transform():
    x = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(log_transform(x), torch.log(x + 1e-8))


def test_log_transform_zeros():
    assert torch.all(torch.isfinite(log_transform(torch.zeros(3))))


def test_standardization():
    result = standardization(torch.tensor([10.0]), train_mean=5.0, train_std=2.0)
    assert abs(result.item() - (5.0 / (2.0 + 1e-8))) < 1e-5


def test_slice_transform():
    x = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    assert slice_transform(x, ymin=1, ymax=3, xmin=0, xmax=4).shape == (2, 4)


def test_pipeline_chained():
    pipeline = build_transform_pipeline(
        [
            {'name': 'sum_normalization'},
            {'name': 'log_transform'},
        ]
    )
    assert torch.all(torch.isfinite(pipeline(SAMPLE_2X2)))


def test_pipeline_params_and_runtime_override():
    pipeline = build_transform_pipeline(
        [
            {'name': 'standardization', 'params': {'train_mean': 0.0, 'train_std': 1.0}},
        ]
    )
    # kwargs override config params
    result = pipeline(torch.tensor([10.0]), train_mean=5.0, train_std=2.0)
    assert abs(result.item() - (5.0 / (2.0 + 1e-8))) < 1e-5


def test_pipeline_empty():
    pipeline = build_transform_pipeline([])
    x = torch.tensor([1.0, 2.0])
    torch.testing.assert_close(pipeline(x), x)
