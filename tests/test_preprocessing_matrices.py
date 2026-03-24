import numpy as np
import pytest

from cfdna.preprocessing.matrices import (
    calculate_min_coverage,
    compute_bin_edges,
    downsample_matrix,
    rebin_matrix,
)


# downsample_matrix
@pytest.mark.parametrize(
    'matrix, target, should_change',
    [
        (np.zeros((10, 10)), 50, False),  # zero matrix
        (np.array([[1, 2], [3, 4]], dtype=np.float64), 100, False),  # sum < target
        (np.array([[1, 2], [3, 4]], dtype=np.float64), 10, False),  # sum == target
    ],
)
def test_downsample_unchanged(matrix, target, should_change):
    result = downsample_matrix(matrix, target)
    np.testing.assert_array_equal(result, matrix)


def test_downsample_sum_matches_target():
    np.random.seed(42)
    m = np.array([[10, 20], [30, 40]], dtype=np.float64)
    result = downsample_matrix(m, 50)
    assert result.sum() == 50
    assert result.shape == m.shape


# rebin_matrix
def test_rebin_even_split():
    m = np.ones((10, 5), dtype=np.float32)
    result = rebin_matrix(m, np.array([0, 5, 10], dtype=np.float64))
    assert result.shape == (2, 5)
    np.testing.assert_array_equal(result[0], np.full(5, 5.0))


def test_rebin_single_bin():
    m = np.arange(30, dtype=np.float32).reshape(6, 5)
    result = rebin_matrix(m, np.array([0, 6], dtype=np.float64))
    np.testing.assert_array_equal(result[0], m.sum(axis=0))


def test_rebin_edges_beyond_matrix():
    m = np.ones((5, 3), dtype=np.float32)
    result = rebin_matrix(m, np.array([0, 3, 10], dtype=np.float64))
    np.testing.assert_array_equal(result[1], np.full(3, 2.0))  # clamped to 5 rows


# compute_bin_edges
def test_compute_bin_edges(sample_matrices):
    edges = compute_bin_edges(sample_matrices, matrix_rows=300, n_rebin_rows=46)
    assert edges[0] == 0
    assert edges[-1] == 300
    assert len(edges) - 1 <= 46


# calculate_min_coverage
def test_calculate_min_coverage(coverage_files):
    assert calculate_min_coverage(coverage_files) == 50
