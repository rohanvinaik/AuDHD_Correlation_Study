"""Data I/O utilities for various formats"""
from pathlib import Path
from typing import Any, Union
import pandas as pd
import numpy as np


def load_data(path: Union[str, Path], **kwargs: Any) -> Union[pd.DataFrame, np.ndarray]:
    """
    Load data from various formats (parquet, hdf5, zarr, csv)

    Args:
        path: Path to data file
        **kwargs: Additional arguments for format-specific loaders

    Returns:
        Loaded data as DataFrame or array
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    elif suffix in [".h5", ".hdf5"]:
        return pd.read_hdf(path, **kwargs)
    elif suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    elif suffix == ".feather":
        return pd.read_feather(path, **kwargs)
    elif suffix == ".zarr":
        import zarr
        return zarr.open(path, mode="r", **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def save_data(
    data: Union[pd.DataFrame, np.ndarray],
    path: Union[str, Path],
    **kwargs: Any
) -> None:
    """
    Save data to various formats

    Args:
        data: Data to save
        path: Output path
        **kwargs: Additional arguments for format-specific writers
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if isinstance(data, pd.DataFrame):
        if suffix == ".parquet":
            data.to_parquet(path, **kwargs)
        elif suffix in [".h5", ".hdf5"]:
            data.to_hdf(path, key="data", **kwargs)
        elif suffix == ".csv":
            data.to_csv(path, **kwargs)
        elif suffix == ".feather":
            data.to_feather(path, **kwargs)
        else:
            raise ValueError(f"Unsupported format for DataFrame: {suffix}")
    elif isinstance(data, np.ndarray):
        if suffix == ".npy":
            np.save(path, data, **kwargs)
        elif suffix == ".zarr":
            import zarr
            zarr.save(str(path), data, **kwargs)
        else:
            raise ValueError(f"Unsupported format for array: {suffix}")
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")