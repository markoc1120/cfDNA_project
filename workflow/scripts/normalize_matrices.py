import numpy as np

from cfdna.preprocessing.matrices import normalize_matrix

if 'snakemake' in globals():
    with open(snakemake.input['sample_cov']) as f:
        sample_total = float(f.read().strip())

    matrix = np.load(snakemake.input['raw'])
    normalized = normalize_matrix(matrix, sample_total)
    np.save(snakemake.output[0], normalized)
