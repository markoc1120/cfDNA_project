import numpy as np

from cfdna.preprocessing.matrices import rebin_matrix

if 'snakemake' in globals():
    bin_edges = np.load(snakemake.input.bin_edges)
    matrix = np.load(snakemake.input.matrix)
    rebinned = rebin_matrix(matrix, bin_edges)
    np.save(snakemake.output[0], rebinned)
