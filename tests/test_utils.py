"""Tests for utility functions"""
import numpy as np
from audhd_correlation.utils.seeds import set_seed


def test_set_seed():
    """Test that set_seed produces reproducible results"""
    set_seed(42)
    result1 = np.random.rand(10)

    set_seed(42)
    result2 = np.random.rand(10)

    np.testing.assert_array_equal(result1, result2)