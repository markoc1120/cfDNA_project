import pytest

from cfdna.preprocessing.dhs import downsample_dhs_files, preprocess_dhs


def run_preprocessor(tmp_path, input_text, matrix_columns=2500):
    inp = tmp_path / 'in.bed'
    out = tmp_path / 'out.bed'
    inp.write_text(input_text)
    preprocess_dhs(str(inp), str(out), matrix_columns)
    return out.read_text()


@pytest.mark.parametrize(
    'input_text, matrix_columns, expected_lines',
    [
        # Non-overlapping: midpoints 1500, 5500, 9500 - all pass
        ('chr1\t1000\t2000\nchr1\t5000\t6000\nchr1\t9000\t10000\n', 2500, 3),
        # Overlapping: midpoints 1500, 2500 - distance 1000 <= 2500
        ('chr1\t1000\t2000\nchr1\t2000\t3000\n', 2500, 1),
        # Same sites but small window - both pass
        ('chr1\t1000\t2000\nchr1\t2000\t3000\n', 500, 2),
        # Different chromosomes - overlap resets
        ('chr1\t1000\t2000\nchr2\t1000\t2000\n', 2500, 2),
        # Empty input
        ('', 2500, 0),
    ],
)
def test_preprocess_dhs(tmp_path, input_text, matrix_columns, expected_lines):
    result = run_preprocessor(tmp_path, input_text, matrix_columns)
    lines = result.strip().split('\n') if result.strip() else []
    assert len(lines) == expected_lines


def test_downsample_to_smallest(make_bed, tmp_path):
    inp_small = make_bed(5)
    inp_large = make_bed(10)
    out1 = str(tmp_path / 'out1.bed')
    out2 = str(tmp_path / 'out2.bed')

    downsample_dhs_files([inp_small, inp_large], [out1, out2])

    for path in [out1, out2]:
        with open(path) as f:
            assert len(f.read().strip().split('\n')) == 5


def test_downsample_smallest_is_copied(make_bed, tmp_path):
    inp_small = make_bed(3)
    inp_large = make_bed(10)
    out1 = str(tmp_path / 'out1.bed')
    out2 = str(tmp_path / 'out2.bed')

    downsample_dhs_files([inp_small, inp_large], [out1, out2])

    with open(inp_small) as f1, open(out1) as f2:
        assert f1.read() == f2.read()
