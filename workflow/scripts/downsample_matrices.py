import numpy as np

from cfdna.preprocessing.matrices import downsample_matrix

if 'snakemake' in globals():
    with open(snakemake.input['mincov']) as f:
        hardcut = int(f.read().strip())

    counts = np.load(snakemake.input['raw'])
    gc_sums = np.load(snakemake.input['gc'])

    downsampled_counts = downsample_matrix(counts, hardcut)

    mean_gc = np.zeros_like(gc_sums, dtype=np.float64)
    nonzero = counts > 0
    mean_gc[nonzero] = gc_sums[nonzero] / counts[nonzero]

    result = downsampled_counts.astype(np.float64) * mean_gc
    np.save(snakemake.output['downsampled'], result)

    with open(snakemake.output['cov'], 'w') as cov_f:
        cov_f.write(str(int(downsampled_counts.sum())))
