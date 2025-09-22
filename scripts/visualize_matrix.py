import numpy as np
import matplotlib.pyplot as plt

from constants import (
    MATRIX_COLUMNS,
    LWPS_UPPER_THRESHOLD,
    LWPS_NUM_POSITIONS
)


def calculate_coverage(matrix: np.ndarray, max_position: int) -> np.ndarray:
    coverage = np.zeros(max_position)
    
    for fragment_length in range(matrix.shape[0]):
        for rel_midpoint in range(matrix.shape[1]):
            count = matrix[fragment_length, rel_midpoint]
            if count > 0:
                # calculate start and end positions from midpoint and length
                start_pos = rel_midpoint - fragment_length // 2
                end_pos = rel_midpoint + fragment_length // 2
                
                # make sure we stay in our boundaries
                start_pos = max(0, start_pos)
                end_pos = min(max_position, end_pos)
                
                # update coverage
                if start_pos < end_pos:
                    coverage[start_pos:end_pos] += count
    
    return coverage


def plot_distributions(
    input_matrix: np.ndarray, 
    output_path: str,
    xlabel: str,
    ylabel: str,
    title: str,
    x_positions: np.ndarray,
):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(x_positions, input_matrix)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    fig.savefig(output_path, dpi=300)


if 'snakemake' in globals():
    matrix_path = snakemake.input.matrix
    lwps_path = snakemake.input.lwps
    
    coverage_plot_path = snakemake.output.coverage_plot
    fragment_lengths_plot_path = snakemake.output.fragment_lengths_plot
    lwps_plot_path = snakemake.output.lwps_plot
    
    with open(matrix_path, 'rb') as f_matrix, open(lwps_path, 'rb') as f_lwps:
        matrix = np.load(f_matrix)
        lwps = np.load(f_lwps)
        
    coverage = calculate_coverage(matrix, MATRIX_COLUMNS)
    plot_distributions(
        coverage,
        coverage_plot_path,
        "Relative midpoint positions",
        "Coverage",
        "Relative midpoint positions VS Coverage",
        np.arange(len(coverage)),
    )
    plot_distributions(
        matrix.sum(axis=1),
        fragment_lengths_plot_path,
        "Fragment lengths",
        "Count",
        "Fragment lengths distribution",
        np.arange(len(matrix)),
    )
    plot_distributions(
        lwps,
        lwps_plot_path,
        "Relative midpoint positions",
        "L-WPS score",
        "Relative midpoint positions VS L-WPS score",
        np.arange(LWPS_UPPER_THRESHOLD, LWPS_NUM_POSITIONS + LWPS_UPPER_THRESHOLD + 1),
    )
    