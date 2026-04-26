from collections import deque

import numpy as np
import pytest


@pytest.fixture
def preprocessor(make_preprocessor):
    return make_preprocessor(
        dhs_text='chr1\t4900\t5100\n',
        frag_lines=[
            'chr1\t4925\t5075\t60\t0.5\n',  # midpoint 5000, length 150 - in window
            'chr1\t4450\t4550\t60\t0.4\n',  # midpoint 4500, length 100 - in window
            'chr1\t100\t250\t60\t0.3\n',  # midpoint 175 - outside window
        ],
    )


def test_read_dhs_to_memory(preprocessor):
    sites, length = preprocessor.read_dhs_to_memory(preprocessor.dhs_files[0])
    assert length == 1
    assert sites[0] == (5000, 'chr1')


def test_get_curr_dhs_empty(preprocessor):
    preprocessor.DHS_sites = [deque()]
    assert preprocessor.get_curr_dhs(0) == (None, None, None)


def test_get_curr_dhs(preprocessor):
    sites, _ = preprocessor.read_dhs_to_memory(preprocessor.dhs_files[0])
    preprocessor.DHS_sites = [sites]
    start, end, chr_ = preprocessor.get_curr_dhs(0)
    assert (start, end, chr_) == (5000 - 1250, 5000 + 1250, 'chr1')


def test_parse_fragment(preprocessor):
    assert preprocessor.parse_fragment('chr1\t100\t250\t60\t0.4\n') == ('chr1', 100, 250, 0.4)


def test_generate_matrix_save(preprocessor):
    result = preprocessor.generate_matrix(should_save=True)
    assert result.shape == (300, 2000)  # 2500 - 2*250


def test_generate_matrix_no_save(preprocessor):
    result = preprocessor.generate_matrix(should_save=False)
    assert result.shape == (300, 2500)


def test_fragment_too_long_skipped(make_preprocessor):
    p = make_preprocessor('chr1\t4900\t5100\n', ['chr1\t4800\t5150\t60\t1\n'])
    assert np.sum(p.generate_matrix(should_save=False)) == 0


def test_invalid_chromosome_skipped(make_preprocessor):
    p = make_preprocessor('chr1\t4900\t5100\n', ['chrM\t4925\t5075\t60\t1\n'])
    assert np.sum(p.generate_matrix(should_save=False)) == 0


def test_multiple_dhs_sites(make_preprocessor):
    p = make_preprocessor(
        'chr1\t4900\t5100\nchr2\t9900\t10100\n',
        ['chr1\t4925\t5075\t60\t1\n', 'chr2\t9925\t10075\t60\t1\n'],
    )
    assert np.sum(p.generate_matrix(should_save=False)) == 2


def test_fragment_past_all_dhs(make_preprocessor):
    p = make_preprocessor(
        'chr1\t100\t200\n',
        ['chr1\t100000\t100150\t60\t1\n', 'chr1\t200000\t200150\t60\t1\n'],
    )
    assert np.sum(p.generate_matrix(should_save=False)) == 0
