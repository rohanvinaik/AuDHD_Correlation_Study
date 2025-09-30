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
        supported = [".parquet", ".h5", ".hdf5", ".csv", ".feather", ".zarr"]
        raise ValueError(
            f"Unsupported file format: '{suffix}'\n\n"
            f"Supported formats for load_data():\n"
            f"  • {', '.join(supported)}\n\n"
            f"Recommendations:\n"
            f"  • For Excel files (.xlsx): Export to CSV first\n"
            f"  • For large datasets: Use Parquet (fast, compressed)\n"
            f"  • For compatibility: Use CSV\n"
            f"  • For arrays: Use Zarr or HDF5"
        )


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
            supported_df = [".parquet", ".h5", ".hdf5", ".csv", ".feather"]
            raise ValueError(
                f"Unsupported format for DataFrame: '{suffix}'\n\n"
                f"Supported formats for saving DataFrames:\n"
                f"  • {', '.join(supported_df)}\n\n"
                f"Recommendations:\n"
                f"  • Best performance: Parquet (.parquet)\n"
                f"  • Best compatibility: CSV (.csv)\n"
                f"  • For Excel: Save as CSV, open in Excel"
            )
    elif isinstance(data, np.ndarray):
        if suffix == ".npy":
            np.save(path, data, **kwargs)
        elif suffix == ".zarr":
            import zarr
            zarr.save(str(path), data, **kwargs)
        else:
            supported_arr = [".npy", ".zarr"]
            raise ValueError(
                f"Unsupported format for NumPy array: '{suffix}'\n\n"
                f"Supported formats for saving arrays:\n"
                f"  • {', '.join(supported_arr)}\n\n"
                f"Recommendations:\n"
                f"  • For NumPy arrays: .npy (fast, native)\n"
                f"  • For large arrays: .zarr (chunked, compressed)\n"
                f"  • To save as table: Convert to DataFrame first"
            )
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")