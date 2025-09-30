"""Missing data imputation"""
from typing import Dict, Any
import numpy as np
from sklearn.impute import KNNImputer
from ..config.schema import AppConfig


def run(X: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
    """Impute missing values in feature matrices"""
    imputed = {}

    for modality, data in X.items():
        if isinstance(data, np.ndarray):
            if cfg.preprocess.imputation == "knn":
                imputer = KNNImputer(n_neighbors=5)
                imputed[modality] = imputer.fit_transform(data)
            else:
                imputed[modality] = data
        else:
            imputed[modality] = data

    return imputed