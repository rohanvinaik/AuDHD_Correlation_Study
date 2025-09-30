"""Consensus clustering with HDBSCAN and topology gaps"""
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering
import hdbscan


def _hdb(emb, params):
    """Run HDBSCAN with given parameters"""
    return hdbscan.HDBSCAN(**params).fit_predict(emb)


def consensus(embeddings: dict, cfg):
    """
    Consensus clustering across multiple embeddings with resampling

    Args:
        embeddings: Dict of embedding matrices (name -> array)
        cfg: AppConfig with cluster.consensus and cluster.clusterers

    Returns:
        labels: Final consensus cluster labels
        coassign: Co-assignment matrix
    """
    coassign = None
    labels_store = {}

    for name, emb in embeddings.items():
        for _ in range(cfg.cluster.consensus["resamples"]):
            idx = np.random.choice(len(emb), len(emb), replace=True)
            lab = _hdb(emb[idx], cfg.cluster.clusterers["hdbscan_main"])
            full = -np.ones(len(emb), int)
            full[idx] = lab
            labels_store.setdefault(name, []).append(full)
            mat = (full[:, None] == full[None, :]) & (full[:, None] >= 0)
            coassign = mat if coassign is None else coassign + mat

    coassign = coassign / (len(embeddings) * cfg.cluster.consensus["resamples"])

    # Spectral clustering on thresholded co-assignment matrix
    sc = SpectralClustering(
        n_clusters=None, affinity="precomputed", assign_labels="kmeans"
    )
    labels = sc.fit_predict(
        (coassign >= cfg.cluster.consensus["threshold"]).astype(float)
    )

    return labels, coassign