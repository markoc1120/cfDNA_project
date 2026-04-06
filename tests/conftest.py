import gzip

import numpy as np
import pytest

SAMPLE_IDS = [f'EE881{i:02d}' for i in range(10)]
DHS_NAME = 'Lymphoid'
REBINNED_SHAPE = (46, 2000)
DOWNSAMPLED_SHAPE = (300, 2000)


@pytest.fixture
def matrix_dir(tmp_path):
    rng = np.random.default_rng(42)

    for sid in SAMPLE_IDS:
        for suffix, shape in [('rebinned', REBINNED_SHAPE), ('downsampled', DOWNSAMPLED_SHAPE)]:
            pos = rng.random(shape, dtype=np.float32)
            neg = rng.random(shape, dtype=np.float32)
            np.save(tmp_path / f'{sid}__{DHS_NAME}_{suffix}.npy', pos)
            np.save(tmp_path / f'{sid}__{DHS_NAME}_negative_{suffix}.npy', neg)

    return str(tmp_path) + '/'


@pytest.fixture
def make_bed(tmp_path):
    counter = [0]

    def make(n_lines, chrom='chr1', spacing=5000):
        counter[0] += 1
        path = tmp_path / f'bed_{counter[0]}.bed'
        with open(path, 'w') as f:
            for i in range(n_lines):
                f.write(f'{chrom}\t{i * spacing}\t{i * spacing + 200}\n')
        return str(path)

    return make


@pytest.fixture
def make_preprocessor(tmp_path):
    from cfdna.preprocessing.fragments import Preprocessor

    counter = [0]

    def make(dhs_text, frag_lines, **kwargs):
        counter[0] += 1
        tag = counter[0]
        dhs = tmp_path / f'dhs_{tag}.bed'
        frags = tmp_path / f'frags_{tag}.gz'
        out = tmp_path / f'matrix_{tag}.npy'
        cov = tmp_path / f'cov_{tag}.txt'

        dhs.write_text(dhs_text)
        with gzip.open(str(frags), 'wt') as f:
            f.writelines(frag_lines)

        defaults = dict(matrix_rows=300, matrix_columns=2500, matrix_shift=250)
        defaults.update(kwargs)
        return Preprocessor(str(frags), str(dhs), str(out), str(cov), **defaults)

    return make


@pytest.fixture
def sample_matrices(tmp_path):
    np.random.seed(42)
    paths = []
    for i in range(3):
        m = np.random.poisson(5, size=(300, 100)).astype(np.float64)
        p = tmp_path / f'matrix_{i}.npy'
        np.save(p, m)
        paths.append(str(p))
    return paths


@pytest.fixture
def coverage_files(tmp_path):
    vals = [100, 50, 200]
    paths = []
    for i, v in enumerate(vals):
        p = tmp_path / f'cov_{i}.txt'
        p.write_text(str(v))
        paths.append(str(p))
    return paths
