#!/usr/bin/env python3
"""
Advanced Network Analysis Methods
Implements Gaussian Graphical Models, partial correlations, and enhanced network metrics
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from sklearn.covariance import GraphicalLassoCV
import logging

logger = logging.getLogger(__name__)


@dataclass
class GGMResult:
    """Results from Gaussian Graphical Model analysis"""
    precision_matrix: np.ndarray
    partial_correlations: np.ndarray
    graph: nx.Graph
    alpha_optimal: float
    feature_names: List[str]
    sparsity: float
    hub_nodes: List[str]
    communities: Dict[int, List[str]]


def construct_gaussian_graphical_model(
    data: pd.DataFrame,
    alpha_range: Optional[np.ndarray] = None,
    cv_folds: int = 5,
    threshold: float = 0.01,
    max_iter: int = 1000
) -> GGMResult:
    """
    Construct Gaussian Graphical Model using graphical LASSO

    GGMs use partial correlations (precision matrix) to identify direct relationships
    while controlling for all other variables. This reveals the conditional independence
    structure of the data.

    Parameters
    ----------
    data : pd.DataFrame
        Data matrix (samples × features)
    alpha_range : np.ndarray, optional
        Range of regularization parameters to test
    cv_folds : int
        Number of cross-validation folds
    threshold : float
        Threshold for edge inclusion (absolute partial correlation)
    max_iter : int
        Maximum iterations for optimization

    Returns
    -------
    GGMResult
        Complete GGM analysis results including graph structure
    """
    logger.info(f"Constructing GGM for {data.shape[1]} features, {data.shape[0]} samples")

    # Default alpha range if not specified
    if alpha_range is None:
        alpha_range = np.logspace(-3, 0, 20)

    # Fit graphical LASSO with cross-validation
    glasso = GraphicalLassoCV(
        alphas=alpha_range,
        cv=cv_folds,
        max_iter=max_iter,
        n_jobs=-1,
        verbose=0
    )

    glasso.fit(data.values)

    # Extract precision matrix (inverse covariance)
    # precision[i,j] = partial correlation controlling for all other variables
    precision_matrix = glasso.precision_

    # Convert precision to partial correlations (normalized)
    # partial_corr[i,j] = -precision[i,j] / sqrt(precision[i,i] * precision[j,j])
    diag = np.sqrt(np.diag(precision_matrix))
    partial_correlations = -precision_matrix / np.outer(diag, diag)
    np.fill_diagonal(partial_correlations, 1.0)

    # Construct network graph
    G = nx.Graph()
    G.add_nodes_from(data.columns)

    n_features = len(data.columns)
    edges_added = 0

    for i in range(n_features):
        for j in range(i + 1, n_features):
            partial_corr = partial_correlations[i, j]
            if abs(partial_corr) > threshold:
                G.add_edge(
                    data.columns[i],
                    data.columns[j],
                    weight=partial_corr,
                    abs_weight=abs(partial_corr)
                )
                edges_added += 1

    logger.info(f"Created graph with {edges_added} edges (threshold={threshold})")

    # Calculate sparsity
    max_edges = n_features * (n_features - 1) / 2
    sparsity = 1 - (edges_added / max_edges)

    # Identify hub nodes (high degree centrality)
    if edges_added > 0:
        degree_centrality = nx.degree_centrality(G)
        hub_threshold = np.percentile(list(degree_centrality.values()), 90)
        hub_nodes = [node for node, cent in degree_centrality.items() if cent >= hub_threshold]

        # Detect communities
        if nx.is_connected(G):
            communities = nx.community.greedy_modularity_communities(G)
        else:
            # Handle disconnected components
            communities = list(nx.connected_components(G))

        community_dict = {i: list(comm) for i, comm in enumerate(communities)}
    else:
        hub_nodes = []
        community_dict = {}

    logger.info(f"Identified {len(hub_nodes)} hub nodes and {len(community_dict)} communities")

    return GGMResult(
        precision_matrix=precision_matrix,
        partial_correlations=partial_correlations,
        graph=G,
        alpha_optimal=glasso.alpha_,
        feature_names=list(data.columns),
        sparsity=sparsity,
        hub_nodes=hub_nodes,
        communities=community_dict
    )


def compare_correlation_vs_partial(
    data: pd.DataFrame,
    ggm_result: GGMResult
) -> pd.DataFrame:
    """
    Compare regular correlations vs partial correlations

    This reveals which correlations are direct vs mediated by other variables.
    Large differences indicate indirect relationships.

    Parameters
    ----------
    data : pd.DataFrame
        Original data
    ggm_result : GGMResult
        Results from GGM analysis

    Returns
    -------
    pd.DataFrame
        Comparison table with correlation, partial correlation, and difference
    """
    # Compute regular correlations
    regular_corr = data.corr().values
    partial_corr = ggm_result.partial_correlations

    comparisons = []
    n = len(ggm_result.feature_names)

    for i in range(n):
        for j in range(i + 1, n):
            comparisons.append({
                'feature_1': ggm_result.feature_names[i],
                'feature_2': ggm_result.feature_names[j],
                'correlation': regular_corr[i, j],
                'partial_correlation': partial_corr[i, j],
                'difference': abs(regular_corr[i, j] - partial_corr[i, j]),
                'is_direct': abs(partial_corr[i, j]) > 0.01,  # Threshold from GGM
                'is_indirect': (abs(regular_corr[i, j]) > 0.3) and (abs(partial_corr[i, j]) < 0.01)
            })

    df = pd.DataFrame(comparisons)
    df = df.sort_values('difference', ascending=False)

    logger.info(f"Found {df['is_direct'].sum()} direct relationships")
    logger.info(f"Found {df['is_indirect'].sum()} indirect relationships")

    return df


def analyze_network_robustness(
    data: pd.DataFrame,
    n_bootstrap: int = 100,
    threshold: float = 0.01
) -> Dict[str, Any]:
    """
    Assess network stability via bootstrap

    Parameters
    ----------
    data : pd.DataFrame
        Original data
    n_bootstrap : int
        Number of bootstrap samples
    threshold : float
        Edge threshold

    Returns
    -------
    Dict containing edge stability and centrality stability
    """
    logger.info(f"Performing {n_bootstrap} bootstrap iterations")

    n_features = len(data.columns)
    edge_counts = np.zeros((n_features, n_features))
    centrality_scores = []

    for i in range(n_bootstrap):
        # Bootstrap sample
        boot_data = data.sample(n=len(data), replace=True)

        # Construct GGM
        try:
            ggm = construct_gaussian_graphical_model(boot_data, threshold=threshold)

            # Count edge occurrences
            for edge in ggm.graph.edges():
                idx1 = data.columns.get_loc(edge[0])
                idx2 = data.columns.get_loc(edge[1])
                edge_counts[idx1, idx2] += 1
                edge_counts[idx2, idx1] += 1

            # Record centrality
            if ggm.graph.number_of_edges() > 0:
                centrality = nx.degree_centrality(ggm.graph)
                centrality_scores.append(centrality)
        except:
            continue

    # Edge stability: proportion of bootstraps where edge appears
    edge_stability = edge_counts / n_bootstrap

    # Centrality stability: coefficient of variation
    if centrality_scores:
        centrality_df = pd.DataFrame(centrality_scores).fillna(0)
        centrality_stability = {
            'mean': centrality_df.mean().to_dict(),
            'std': centrality_df.std().to_dict(),
            'cv': (centrality_df.std() / centrality_df.mean()).to_dict()
        }
    else:
        centrality_stability = {}

    return {
        'edge_stability': edge_stability,
        'edge_stability_mean': edge_stability.mean(),
        'centrality_stability': centrality_stability,
        'n_bootstrap': n_bootstrap
    }


def identify_mediating_paths(
    ggm_result: GGMResult,
    source: str,
    target: str,
    max_path_length: int = 3
) -> List[List[str]]:
    """
    Find all mediating paths between two nodes

    In GGMs, absence of direct edge but presence of paths
    indicates mediation by intermediate variables.

    Parameters
    ----------
    ggm_result : GGMResult
        GGM analysis results
    source : str
        Source node
    target : str
        Target node
    max_path_length : int
        Maximum path length to consider

    Returns
    -------
    List of paths (each path is list of nodes)
    """
    G = ggm_result.graph

    if not G.has_node(source) or not G.has_node(target):
        return []

    # Find all simple paths
    try:
        all_paths = nx.all_simple_paths(
            G, source, target,
            cutoff=max_path_length
        )
        paths = list(all_paths)
    except nx.NetworkXNoPath:
        paths = []

    # Filter: only return paths of length > 1 (mediating paths)
    mediating_paths = [p for p in paths if len(p) > 2]

    return mediating_paths


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Generate test data
    np.random.seed(42)
    n_samples = 200

    # Create data with known structure:
    # X1 -> X2 -> X3 (chain)
    # X4 -> X2, X4 -> X3 (common cause)
    X1 = np.random.randn(n_samples)
    X2 = 0.7 * X1 + np.random.randn(n_samples) * 0.5
    X4 = np.random.randn(n_samples)
    X3 = 0.5 * X2 + 0.4 * X4 + np.random.randn(n_samples) * 0.5
    X5 = np.random.randn(n_samples)  # Independent

    data = pd.DataFrame({
        'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5
    })

    # Fit GGM
    ggm = construct_gaussian_graphical_model(data, threshold=0.1)

    print(f"\nGGM Results:")
    print(f"  Optimal alpha: {ggm.alpha_optimal:.4f}")
    print(f"  Sparsity: {ggm.sparsity:.2%}")
    print(f"  Number of edges: {ggm.graph.number_of_edges()}")
    print(f"  Hub nodes: {ggm.hub_nodes}")

    # Compare correlations
    comparison = compare_correlation_vs_partial(data, ggm)
    print(f"\nTop 5 differences (correlation vs partial):")
    print(comparison.head())
