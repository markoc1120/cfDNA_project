import numpy as np

def rebin_matrix(matrix, bin_edges):
    n_bins = len(bin_edges) - 1
    rebinned = np.zeros((n_bins, matrix.shape[1]), dtype=np.float32)
    for i in range(n_bins):
        low  = int(np.ceil(bin_edges[i]))
        high = int(np.ceil(bin_edges[i + 1]))
        high = min(high, matrix.shape[0])
        if low < high:
            rebinned[i] = matrix[low:high, :].sum(axis=0)
    return rebinned

if 'snakemake' in globals():
    bin_edges = np.load(snakemake.input.bin_edges)
    matrix    = np.load(snakemake.input.matrix)
    rebinned  = rebin_matrix(matrix, bin_edges)
    np.save(snakemake.output[0], rebinned)
