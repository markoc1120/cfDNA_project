import numpy as np
import pandas as pd


def downsample_matrix(matrix: np.ndarray, target_sum: int) -> np.ndarray:
    current_sum = np.sum(matrix)

    if current_sum == 0:
        return matrix
    if current_sum <= target_sum:
        return matrix

    flat_matrix = matrix.flatten()
    probabilities = flat_matrix / current_sum

    downsampled_flat = np.random.multinomial(target_sum, probabilities)
    return downsampled_flat.reshape(matrix.shape)


def rebin_matrix(matrix: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    n_bins = len(bin_edges) - 1
    rebinned = np.zeros((n_bins, matrix.shape[1]), dtype=np.float32)
    for i in range(n_bins):
        low = int(np.ceil(bin_edges[i]))
        high = int(np.ceil(bin_edges[i + 1]))
        high = min(high, matrix.shape[0])
        if low < high:
            rebinned[i] = matrix[low:high, :].sum(axis=0)
    return rebinned


def compute_bin_edges(matrix_paths: list[str], matrix_rows: int, n_rebin_rows: int) -> np.ndarray:
    row_sums = np.zeros(matrix_rows)
    total = len(matrix_paths)
    for i, path in enumerate(matrix_paths):
        if not i % 50:
            print(f'{i}/{total} (row_sums)')
        row_sums += np.load(path).sum(axis=1)

    lengths_expanded = np.repeat(np.arange(matrix_rows), row_sums.astype(np.int64))
    _, bin_edges = pd.qcut(lengths_expanded, q=n_rebin_rows, retbins=True, duplicates='drop')
    bin_edges[0] = 0
    bin_edges[-1] = matrix_rows
    print(f'bin edges ({len(bin_edges) - 1} bins): {bin_edges}')
    return bin_edges


def calculate_min_coverage(cov_files: list[str]) -> int:
    vals = []
    for cov in cov_files:
        with open(cov) as f:
            vals.append(int(f.read().strip()))
    return min(vals)
