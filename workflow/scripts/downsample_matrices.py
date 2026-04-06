import numpy as np

from cfdna.preprocessing.matrices import downsample_matrix

if 'snakemake' in globals():
    with open(snakemake.input['mincov']) as f:
        hardcut = int(f.read().strip())

    matrix = np.load(snakemake.input['raw'])
    downsampled = downsample_matrix(matrix, hardcut)
    np.save(snakemake.output[0], downsampled)
