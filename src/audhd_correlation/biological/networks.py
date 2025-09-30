"""Network analysis for metabolic pathways and protein-protein interactions

Reconstructs networks and identifies key regulatory nodes.
"""
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler


@dataclass
class NetworkResult:
    """Result of network analysis"""
    # Network
    graph: nx.Graph
    n_nodes: int
    n_edges: int

    # Centrality measures
    degree_centrality: Dict[str, float]
    betweenness_centrality: Dict[str, float]
    closeness_centrality: Dict[str, float]
    eigenvector_centrality: Dict[str, float]

    # Hub nodes
    hub_nodes: List[str]
    hub_threshold: float

    # Communities
    communities: Optional[List[Set[str]]] = None
    modularity: Optional[float] = None

    # Network statistics
    avg_degree: Optional[float] = None
    density: Optional[float] = None
    avg_clustering: Optional[float] = None
    avg_path_length: Optional[float] = None

    def top_hubs(self, n: int = 10, metric: str = 'degree') -> List[Tuple[str, float]]:
        """Return top N hub nodes by centrality metric"""
        if metric == 'degree':
            centrality = self.degree_centrality
        elif metric == 'betweenness':
            centrality = self.betweenness_centrality
        elif metric == 'closeness':
            centrality = self.closeness_centrality
        elif metric == 'eigenvector':
            centrality = self.eigenvector_centrality
        else:
            raise ValueError(f"Unknown metric: {metric}")

        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:n]


def reconstruct_metabolic_network(
    metabolite_data: pd.DataFrame,
    correlation_threshold: float = 0.6,
    method: str = 'pearson',
    min_samples: int = 10,
) -> NetworkResult:
    """
    Reconstruct metabolic network from metabolomics data

    Args:
        metabolite_data: Metabolite abundance matrix (samples × metabolites)
        correlation_threshold: Minimum correlation for edge
        method: Correlation method ('pearson', 'spearman')
        min_samples: Minimum samples required

    Returns:
        NetworkResult with metabolic network
    """
    if len(metabolite_data) < min_samples:
        raise ValueError(f"Need at least {min_samples} samples")

    # Calculate correlation matrix
    if method == 'pearson':
        corr_matrix = metabolite_data.corr(method='pearson')
    elif method == 'spearman':
        corr_matrix = metabolite_data.corr(method='spearman')
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create graph from significant correlations
    G = nx.Graph()

    metabolites = metabolite_data.columns.tolist()
    G.add_nodes_from(metabolites)

    # Add edges for significant correlations
    n_metabolites = len(metabolites)
    for i in range(n_metabolites):
        for j in range(i + 1, n_metabolites):
            corr_val = corr_matrix.iloc[i, j]

            if abs(corr_val) >= correlation_threshold:
                G.add_edge(
                    metabolites[i],
                    metabolites[j],
                    weight=abs(corr_val),
                    correlation=corr_val,
                )

    # Calculate network properties
    result = _analyze_network(G, hub_threshold=0.7)

    return result


def analyze_ppi_network(
    proteins: List[str],
    ppi_database: Optional[pd.DataFrame] = None,
    expression_data: Optional[pd.DataFrame] = None,
    weight_by_expression: bool = True,
) -> NetworkResult:
    """
    Analyze protein-protein interaction network

    Args:
        proteins: List of proteins of interest
        ppi_database: PPI database (protein1, protein2, score)
        expression_data: Expression data to weight edges
        weight_by_expression: Whether to weight by expression correlation

    Returns:
        NetworkResult with PPI network
    """
    # Load PPI database
    if ppi_database is None:
        ppi_database = _get_example_ppi_database()

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(proteins)

    # Add edges from PPI database
    for _, row in ppi_database.iterrows():
        protein1 = row['protein1']
        protein2 = row['protein2']

        if protein1 in proteins and protein2 in proteins:
            score = row.get('score', 1.0)

            # Weight by expression correlation if available
            if weight_by_expression and expression_data is not None:
                if protein1 in expression_data.columns and protein2 in expression_data.columns:
                    expr_corr = expression_data[protein1].corr(expression_data[protein2])
                    weight = score * abs(expr_corr)
                else:
                    weight = score
            else:
                weight = score

            G.add_edge(protein1, protein2, weight=weight, ppi_score=score)

    # Analyze network
    result = _analyze_network(G, hub_threshold=0.6)

    return result


def _get_example_ppi_database() -> pd.DataFrame:
    """Get example PPI database for demonstration"""
    # Example interactions
    interactions = [
        # Immune system
        ('IL6', 'IL6R', 0.95),
        ('IL6', 'JAK1', 0.90),
        ('JAK1', 'STAT3', 0.95),
        ('TNF', 'TNFRSF1A', 0.95),
        ('TNF', 'NFKB1', 0.85),
        ('IL1B', 'IL1R1', 0.95),
        ('IL1B', 'NFKB1', 0.80),
        # Neurotransmitter system
        ('SLC17A7', 'GRIN1', 0.70),
        ('GRIN1', 'GRIN2A', 0.90),
        ('GRIN1', 'DLG4', 0.85),
        ('DLG4', 'SYN1', 0.75),
        ('DRD2', 'GNAL', 0.80),
        ('HTR2A', 'ARRB2', 0.70),
        # Signal transduction
        ('MTOR', 'RPS6KB1', 0.95),
        ('RPS6KB1', 'RPS6', 0.90),
        ('MTOR', 'EIF4E', 0.85),
    ]

    df = pd.DataFrame(interactions, columns=['protein1', 'protein2', 'score'])
    return df


def _analyze_network(G: nx.Graph, hub_threshold: float = 0.7) -> NetworkResult:
    """Analyze network properties and identify hubs"""

    if len(G.nodes()) == 0:
        # Empty network
        return NetworkResult(
            graph=G,
            n_nodes=0,
            n_edges=0,
            degree_centrality={},
            betweenness_centrality={},
            closeness_centrality={},
            eigenvector_centrality={},
            hub_nodes=[],
            hub_threshold=hub_threshold,
        )

    # Centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # Handle disconnected components for closeness
    if nx.is_connected(G):
        closeness_centrality = nx.closeness_centrality(G)
    else:
        # Calculate for each component
        closeness_centrality = {}
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            closeness = nx.closeness_centrality(subgraph)
            closeness_centrality.update(closeness)

    # Eigenvector centrality (may fail for disconnected graphs)
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
        # Fall back to PageRank
        eigenvector_centrality = nx.pagerank(G)

    # Identify hub nodes (high degree centrality)
    hub_nodes = [
        node for node, centrality in degree_centrality.items()
        if centrality >= hub_threshold
    ]

    # Network statistics
    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees) if degrees else 0.0

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    max_edges = n_nodes * (n_nodes - 1) / 2
    density = n_edges / max_edges if max_edges > 0 else 0.0

    avg_clustering = nx.average_clustering(G)

    # Average path length (only for connected graphs)
    if nx.is_connected(G):
        avg_path_length = nx.average_shortest_path_length(G)
    else:
        # Average over connected components
        lengths = []
        for component in nx.connected_components(G):
            if len(component) > 1:
                subgraph = G.subgraph(component)
                lengths.append(nx.average_shortest_path_length(subgraph))
        avg_path_length = np.mean(lengths) if lengths else float('inf')

    return NetworkResult(
        graph=G,
        n_nodes=n_nodes,
        n_edges=n_edges,
        degree_centrality=degree_centrality,
        betweenness_centrality=betweenness_centrality,
        closeness_centrality=closeness_centrality,
        eigenvector_centrality=eigenvector_centrality,
        hub_nodes=hub_nodes,
        hub_threshold=hub_threshold,
        avg_degree=float(avg_degree),
        density=float(density),
        avg_clustering=float(avg_clustering),
        avg_path_length=float(avg_path_length),
    )


def find_hub_nodes(
    network_result: NetworkResult,
    n_hubs: int = 10,
    metric: str = 'degree',
) -> List[str]:
    """
    Find hub nodes in network

    Args:
        network_result: NetworkResult from network analysis
        n_hubs: Number of hubs to return
        metric: Centrality metric ('degree', 'betweenness', 'closeness', 'eigenvector')

    Returns:
        List of hub node names
    """
    top_hubs = network_result.top_hubs(n=n_hubs, metric=metric)
    return [node for node, _ in top_hubs]


def community_detection(
    network_result: NetworkResult,
    method: str = 'louvain',
    resolution: float = 1.0,
) -> NetworkResult:
    """
    Detect communities in network

    Args:
        network_result: NetworkResult from network analysis
        method: Community detection method ('louvain', 'greedy', 'label_propagation')
        resolution: Resolution parameter for Louvain

    Returns:
        NetworkResult with communities added
    """
    G = network_result.graph

    if len(G.nodes()) == 0:
        network_result.communities = []
        network_result.modularity = 0.0
        return network_result

    # Detect communities
    if method == 'louvain':
        communities = _louvain_communities(G, resolution=resolution)
    elif method == 'greedy':
        communities = list(nx.community.greedy_modularity_communities(G))
    elif method == 'label_propagation':
        communities = list(nx.community.label_propagation_communities(G))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Calculate modularity
    modularity = nx.community.modularity(G, communities)

    # Update result
    network_result.communities = communities
    network_result.modularity = float(modularity)

    return network_result


def _louvain_communities(G: nx.Graph, resolution: float = 1.0) -> List[Set[str]]:
    """
    Louvain community detection

    Simplified implementation. In production, use python-louvain package.
    """
    # For now, use greedy modularity as fallback
    # Full implementation would use: import community; community.best_partition(G)
    communities = list(nx.community.greedy_modularity_communities(G))
    return communities


def visualize_network(
    network_result: NetworkResult,
    layout: str = 'spring',
    node_size_by: str = 'degree',
    color_by_community: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize network

    Args:
        network_result: NetworkResult to visualize
        layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
        node_size_by: Size nodes by centrality ('degree', 'betweenness', 'closeness')
        color_by_community: Color nodes by community
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to show figure
    """
    import matplotlib.pyplot as plt

    G = network_result.graph

    if len(G.nodes()) == 0:
        warnings.warn("Empty network, nothing to visualize")
        return

    # Create layout
    if layout == 'spring':
        pos = nx.spring_layout(G, k=0.5, iterations=50)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)

    # Node sizes
    if node_size_by == 'degree':
        centrality = network_result.degree_centrality
    elif node_size_by == 'betweenness':
        centrality = network_result.betweenness_centrality
    elif node_size_by == 'closeness':
        centrality = network_result.closeness_centrality
    else:
        centrality = {node: 1.0 for node in G.nodes()}

    node_sizes = [centrality.get(node, 0) * 1000 + 100 for node in G.nodes()]

    # Node colors
    if color_by_community and network_result.communities:
        node_colors = []
        node_to_community = {}

        for i, community in enumerate(network_result.communities):
            for node in community:
                node_to_community[node] = i

        node_colors = [node_to_community.get(node, -1) for node in G.nodes()]
    else:
        node_colors = 'skyblue'

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw network
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.Set3,
        alpha=0.8,
        ax=ax,
    )

    nx.draw_networkx_edges(
        G, pos,
        alpha=0.3,
        ax=ax,
    )

    # Draw labels for hub nodes
    hub_labels = {node: node for node in network_result.hub_nodes}
    nx.draw_networkx_labels(
        G, pos,
        labels=hub_labels,
        font_size=10,
        font_weight='bold',
        ax=ax,
    )

    ax.set_title(
        f"Network: {network_result.n_nodes} nodes, {network_result.n_edges} edges\n"
        f"Density: {network_result.density:.3f}, "
        f"Avg clustering: {network_result.avg_clustering:.3f}",
        fontsize=14,
    )
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def compare_networks(
    network1: NetworkResult,
    network2: NetworkResult,
    name1: str = 'Network 1',
    name2: str = 'Network 2',
) -> pd.DataFrame:
    """
    Compare two networks

    Args:
        network1: First network
        network2: Second network
        name1: Name for first network
        name2: Name for second network

    Returns:
        DataFrame with comparison statistics
    """
    stats = {
        'Metric': [
            'Number of nodes',
            'Number of edges',
            'Density',
            'Average degree',
            'Average clustering',
            'Average path length',
            'Number of hubs',
            'Modularity',
        ],
        name1: [
            network1.n_nodes,
            network1.n_edges,
            network1.density,
            network1.avg_degree,
            network1.avg_clustering,
            network1.avg_path_length,
            len(network1.hub_nodes),
            network1.modularity if network1.modularity else np.nan,
        ],
        name2: [
            network2.n_nodes,
            network2.n_edges,
            network2.density,
            network2.avg_degree,
            network2.avg_clustering,
            network2.avg_path_length,
            len(network2.hub_nodes),
            network2.modularity if network2.modularity else np.nan,
        ],
    }

    df = pd.DataFrame(stats)
    return df