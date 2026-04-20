import numpy as np

from cfdna.preprocessing.matrices import compute_bin_edges

if 'snakemake' in globals():
    matrix_paths = snakemake.input.matrices
    bin_edges = compute_bin_edges(
        matrix_paths,
        matrix_rows=snakemake.params.matrix_rows,
    )
    np.save(snakemake.output.bin_edges, bin_edges)
