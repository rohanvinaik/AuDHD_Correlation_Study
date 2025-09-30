"""Tests for preprocessing modules"""
import numpy as np
from src.audhd_correlation.preprocess.impute import run as impute_run


def test_impute_shapes():
    X = {"metabolomic": np.array([[1.0, np.nan], [2.0, 3.0]])}
    cfg = type("obj", (), {"preprocess": type("P", (), {"imputation": "knn"})})
    out = impute_run(X, cfg)
    assert out["metabolomic"].shape == (2, 2)
    assert np.isfinite(out["metabolomic"]).all()