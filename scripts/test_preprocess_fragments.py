import pytest
import tempfile
import gzip
import numpy as np
import sys
import os
from unittest.mock import patch

class MockConstants:
    MATRIX_ROWS = 5
    MATRIX_COLUMNS = 10

sys.modules['constants'] = MockConstants()


from .preprocessing import Preprocessor

# TODO: fix tests

def test_preprocessing():
    with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as dhs_f:
        with gzip.open(dhs_f.name, 'wt') as f:
            f.write("100\t200\n")  # DHS site, midpoint=150, window=145-155
        dhs_file = dhs_f.name
    
    with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as frag_f:
        with gzip.open(frag_f.name, 'wt') as f:
            f.write("chr1\t147\t150\n")  # fragment length=4, positions 2-6 in window
            f.write("chr1\t148\t151\n")  # fragment length=4, positions 3-7 in window
        frag_file = frag_f.name
    
    p = Preprocessor(frag_file, dhs_file, "out.npy")
    result = p.generate_matrix(should_save=False)
    
    assert result.shape == (1, 5, 10)  # 1 DHS, 5 rows, 10 columns
    assert result[0, 3, 2:6].sum() == 7  # fragment length 4, positions 2-6 + fragment length 4, positions 3-6
    
    os.unlink(dhs_file)
    os.unlink(frag_file)

def test_fragment_outside_window():
    with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as dhs_f:
        with gzip.open(dhs_f.name, 'wt') as f:
            f.write("100\t200\n")  # midpoint=150, window=145-155
        dhs_file = dhs_f.name
    
    with tempfile.NamedTemporaryFile(suffix='.gz', delete=False) as frag_f:
        with gzip.open(frag_f.name, 'wt') as f:
            f.write("chr1\t50\t60\n")  # outside window
        frag_file = frag_f.name
    
    p = Preprocessor(frag_file, dhs_file, "out.npy")
    result = p.generate_matrix(should_save=False)
    
    assert result.sum() == 0  # no overlap = no counts
    
    os.unlink(dhs_file)
    os.unlink(frag_file)

def test_parse_fragment_simple(): 
    p = Preprocessor("", "", "")
    start, end = p.parse_fragment("chr1\t1000\t1050")
    
    assert start == 1000
    assert end == 1050