import numpy as np
import pandas as pd
import yaml
import os

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

MATRIX_ROWS = config['matrix_rows']
N_REBIN_ROWS = config['n_rebin_rows']

if 'snakemake' in globals():
    matrix_paths = snakemake.input.matrices
    out_path = snakemake.output.bin_edges

    row_sums = np.zeros(MATRIX_ROWS)
    total = len(matrix_paths)
    for i, path in enumerate(matrix_paths):
        if not i % 50:
            print(f'{i}/{total} (row_sums)')
        row_sums += np.load(path).sum(axis=1)

    lengths_expanded = np.repeat(np.arange(MATRIX_ROWS), row_sums.astype(np.int64))
    _, bin_edges = pd.qcut(lengths_expanded, q=N_REBIN_ROWS, retbins=True, duplicates='drop')
    bin_edges[0]  = 0
    bin_edges[-1] = MATRIX_ROWS
    print(f'bin edges ({len(bin_edges)-1} bins): {bin_edges}')
    np.save(out_path, bin_edges)
