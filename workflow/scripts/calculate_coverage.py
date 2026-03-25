import numpy as np

if 'snakemake' in globals():
    matrix = np.load(snakemake.input[0])
    total_cov = np.sum(matrix)
    with open(snakemake.output[0], 'w') as cov_f:
        cov_f.write(str(int(total_cov)))
