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
    # row wise binning
    n_bins = len(bin_edges) - 1
    rebinned_rows = np.zeros((n_bins, matrix.shape[1]), dtype=np.float32)
    for i in range(n_bins):
        low = int(np.ceil(bin_edges[i]))
        high = int(np.ceil(bin_edges[i + 1]))
        high = min(high, matrix.shape[0])
        if low < high:
            rebinned_rows[i] = matrix[low:high, :].sum(axis=0)

    # column wise binning + slice off 4 at each end (200 -> 192)
    rebinned = rebinned_rows.reshape(rebinned_rows.shape[0], -1, 10).sum(axis=2)
    return rebinned[:, 4:-4]


def compute_bin_edges(matrix_paths: list[str], matrix_rows: int, divisor: int = 8) -> np.ndarray:
    row_sums = np.zeros(matrix_rows)
    total = len(matrix_paths)
    for i, path in enumerate(matrix_paths):
        if not i % 50:
            print(f'{i}/{total} (row_sums)')
        row_sums += np.load(path).sum(axis=1)

    lengths_expanded = np.repeat(np.arange(matrix_rows), row_sums.astype(np.int64))

    start, end = 4, 13
    for multiplier in range(start, end):
        q = multiplier * divisor
        _, edges = pd.qcut(lengths_expanded, q=q, retbins=True, duplicates='drop')
        n = len(edges) - 1
        if n % divisor == 0:
            edges[0] = 0
            edges[-1] = matrix_rows
            print(f'bin edges ({n} bins, q={q}): {edges}')
            return edges

    raise RuntimeError(
        f'no q in [{start * divisor},{(end - 1) * divisor}] produced a bin count '
        f'divisible by {divisor}'
    )


def calculate_min_coverage(cov_files: list[str]) -> float:
    vals = []
    for cov in cov_files:
        with open(cov) as f:
            vals.append(float(f.read().strip()))
    return min(vals)
