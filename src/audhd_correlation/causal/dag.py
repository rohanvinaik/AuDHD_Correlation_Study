"""Directed Acyclic Graph (DAG) construction and validation

Tools for building causal DAGs from domain knowledge and identifying
confounders, mediators, and colliders.
"""
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


@dataclass
class DAGSpecification:
    """Specification of a causal DAG"""
    nodes: List[str]
    edges: List[Tuple[str, str]]
    exposure: str
    outcome: str
    confounders: Optional[List[str]] = None
    mediators: Optional[List[str]] = None
    colliders: Optional[List[str]] = None
    instruments: Optional[List[str]] = None


def build_dag(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    exposure: str,
    outcome: str,
    validate: bool = True,
) -> nx.DiGraph:
    """
    Build directed acyclic graph from nodes and edges

    Args:
        nodes: List of variable names
        edges: List of (source, target) tuples
        exposure: Name of exposure variable
        outcome: Name of outcome variable
        validate: Check if DAG is acyclic

    Returns:
        NetworkX DiGraph
    """
    # Create directed graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Store metadata
    G.graph['exposure'] = exposure
    G.graph['outcome'] = outcome

    if validate:
        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Graph contains cycles - not a valid DAG")

        if exposure not in G.nodes:
            raise ValueError(f"Exposure '{exposure}' not in DAG")

        if outcome not in G.nodes:
            raise ValueError(f"Outcome '{outcome}' not in DAG")

    return G


def validate_dag(G: nx.DiGraph) -> Dict[str, bool]:
    """
    Validate DAG structure and identify potential issues

    Args:
        G: NetworkX DiGraph

    Returns:
        Dictionary of validation checks
    """
    checks = {}

    # Check if acyclic
    checks['is_acyclic'] = nx.is_directed_acyclic_graph(G)

    # Check if connected
    checks['is_weakly_connected'] = nx.is_weakly_connected(G)

    # Check for isolated nodes
    isolated = list(nx.isolates(G))
    checks['has_isolated_nodes'] = len(isolated) > 0
    checks['isolated_nodes'] = isolated

    # Check if exposure affects outcome
    exposure = G.graph.get('exposure')
    outcome = G.graph.get('outcome')

    if exposure and outcome:
        checks['exposure_affects_outcome'] = nx.has_path(G, exposure, outcome)

    # Check for colliders on exposure-outcome path
    if exposure and outcome:
        try:
            paths = list(nx.all_simple_paths(G, exposure, outcome))
            checks['n_paths'] = len(paths)
        except nx.NetworkXNoPath:
            checks['n_paths'] = 0

    return checks


def identify_confounders(
    G: nx.DiGraph,
    exposure: Optional[str] = None,
    outcome: Optional[str] = None,
) -> Set[str]:
    """
    Identify confounders of exposure-outcome relationship

    A confounder is a common cause of both exposure and outcome.

    Args:
        G: NetworkX DiGraph
        exposure: Exposure variable (if None, use G.graph['exposure'])
        outcome: Outcome variable (if None, use G.graph['outcome'])

    Returns:
        Set of confounder variable names
    """
    if exposure is None:
        exposure = G.graph.get('exposure')
    if outcome is None:
        outcome = G.graph.get('outcome')

    if not exposure or not outcome:
        raise ValueError("Exposure and outcome must be specified")

    confounders = set()

    # A confounder has directed paths to both exposure and outcome
    for node in G.nodes():
        if node == exposure or node == outcome:
            continue

        has_path_to_exposure = nx.has_path(G, node, exposure)
        has_path_to_outcome = nx.has_path(G, node, outcome)

        if has_path_to_exposure and has_path_to_outcome:
            confounders.add(node)

    return confounders


def identify_mediators(
    G: nx.DiGraph,
    exposure: Optional[str] = None,
    outcome: Optional[str] = None,
) -> Set[str]:
    """
    Identify mediators on exposure-outcome path

    A mediator lies on a causal path from exposure to outcome.

    Args:
        G: NetworkX DiGraph
        exposure: Exposure variable
        outcome: Outcome variable

    Returns:
        Set of mediator variable names
    """
    if exposure is None:
        exposure = G.graph.get('exposure')
    if outcome is None:
        outcome = G.graph.get('outcome')

    if not exposure or not outcome:
        raise ValueError("Exposure and outcome must be specified")

    mediators = set()

    # Find all paths from exposure to outcome
    try:
        paths = list(nx.all_simple_paths(G, exposure, outcome))

        # Nodes on these paths (excluding exposure and outcome) are potential mediators
        for path in paths:
            for node in path[1:-1]:  # Exclude first (exposure) and last (outcome)
                mediators.add(node)

    except nx.NetworkXNoPath:
        pass

    return mediators


def identify_colliders(
    G: nx.DiGraph,
    exposure: Optional[str] = None,
    outcome: Optional[str] = None,
) -> Set[str]:
    """
    Identify colliders that should NOT be conditioned on

    A collider is a node with two or more incoming edges.
    Conditioning on colliders can introduce bias.

    Args:
        G: NetworkX DiGraph
        exposure: Exposure variable
        outcome: Outcome variable

    Returns:
        Set of collider variable names
    """
    colliders = set()

    for node in G.nodes():
        # A collider has 2+ parents
        parents = list(G.predecessors(node))
        if len(parents) >= 2:
            colliders.add(node)

    return colliders


def identify_adjustment_set(
    G: nx.DiGraph,
    exposure: Optional[str] = None,
    outcome: Optional[str] = None,
    method: str = 'backdoor',
) -> Set[str]:
    """
    Identify variables to adjust for to estimate causal effect

    Args:
        G: NetworkX DiGraph
        exposure: Exposure variable
        outcome: Outcome variable
        method: 'backdoor', 'minimal', or 'all_confounders'

    Returns:
        Set of variables to adjust for
    """
    if exposure is None:
        exposure = G.graph.get('exposure')
    if outcome is None:
        outcome = G.graph.get('outcome')

    if method == 'all_confounders':
        return identify_confounders(G, exposure, outcome)

    elif method == 'backdoor':
        # Identify backdoor paths (paths from exposure to outcome through confounders)
        # Block these by adjusting for confounders
        confounders = identify_confounders(G, exposure, outcome)
        mediators = identify_mediators(G, exposure, outcome)
        colliders = identify_colliders(G)

        # Adjust for confounders, not mediators or colliders
        adjustment_set = confounders - mediators - colliders

        return adjustment_set

    elif method == 'minimal':
        # Find minimal adjustment set
        confounders = identify_confounders(G, exposure, outcome)
        colliders = identify_colliders(G)

        # Remove colliders
        adjustment_set = confounders - colliders

        return adjustment_set

    else:
        raise ValueError(f"Unknown method: {method}")


def plot_dag(
    G: nx.DiGraph,
    figsize: Tuple[int, int] = (12, 8),
    node_color_map: Optional[Dict[str, str]] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize DAG

    Args:
        G: NetworkX DiGraph
        figsize: Figure size
        node_color_map: Dictionary mapping node types to colors
        show: Show plot
        save_path: Path to save figure

    Returns:
        Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get exposure and outcome
    exposure = G.graph.get('exposure')
    outcome = G.graph.get('outcome')

    # Identify node types
    confounders = identify_confounders(G)
    mediators = identify_mediators(G)
    colliders = identify_colliders(G)

    # Default color map
    if node_color_map is None:
        node_color_map = {
            'exposure': '#ff6b6b',      # Red
            'outcome': '#4ecdc4',       # Teal
            'confounder': '#feca57',    # Yellow
            'mediator': '#48dbfb',      # Blue
            'collider': '#ff9ff3',      # Pink
            'other': '#95afc0',         # Gray
        }

    # Assign colors to nodes
    node_colors = []
    for node in G.nodes():
        if node == exposure:
            node_colors.append(node_color_map['exposure'])
        elif node == outcome:
            node_colors.append(node_color_map['outcome'])
        elif node in confounders:
            node_colors.append(node_color_map['confounder'])
        elif node in mediators:
            node_colors.append(node_color_map['mediator'])
        elif node in colliders:
            node_colors.append(node_color_map['collider'])
        else:
            node_colors.append(node_color_map['other'])

    # Layout
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        pos = nx.circular_layout(G)

    # Draw
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='#34495e', arrows=True,
                          arrowsize=20, arrowstyle='->', width=2, ax=ax)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_color_map['exposure'],
                  markersize=10, label='Exposure'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_color_map['outcome'],
                  markersize=10, label='Outcome'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_color_map['confounder'],
                  markersize=10, label='Confounder'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_color_map['mediator'],
                  markersize=10, label='Mediator'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_color_map['collider'],
                  markersize=10, label='Collider'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    ax.set_title('Causal DAG', fontsize=14, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def create_audhd_dag() -> nx.DiGraph:
    """
    Create example DAG for AuDHD correlation study

    Returns:
        NetworkX DiGraph with AuDHD variables
    """
    nodes = [
        'genetics',           # Genetic variants
        'environment',        # Environmental exposures
        'brain_structure',    # Brain structural measures
        'neurotransmitters',  # Neurotransmitter levels
        'gut_microbiome',     # Gut microbiome composition
        'inflammation',       # Inflammatory markers
        'adhd_symptoms',      # ADHD symptom severity
        'asd_symptoms',       # ASD symptom severity
        'comorbidity',        # AuDHD comorbidity status
    ]

    edges = [
        # Genetic effects
        ('genetics', 'brain_structure'),
        ('genetics', 'neurotransmitters'),
        ('genetics', 'adhd_symptoms'),
        ('genetics', 'asd_symptoms'),

        # Environmental effects
        ('environment', 'gut_microbiome'),
        ('environment', 'inflammation'),
        ('environment', 'brain_structure'),

        # G×E interactions (simplified)
        ('gut_microbiome', 'inflammation'),
        ('gut_microbiome', 'neurotransmitters'),

        # Brain and biological pathways
        ('brain_structure', 'neurotransmitters'),
        ('inflammation', 'brain_structure'),
        ('inflammation', 'neurotransmitters'),

        # Symptom pathways
        ('neurotransmitters', 'adhd_symptoms'),
        ('neurotransmitters', 'asd_symptoms'),
        ('brain_structure', 'adhd_symptoms'),
        ('brain_structure', 'asd_symptoms'),

        # Comorbidity
        ('adhd_symptoms', 'comorbidity'),
        ('asd_symptoms', 'comorbidity'),
    ]

    G = build_dag(
        nodes=nodes,
        edges=edges,
        exposure='genetics',
        outcome='comorbidity',
    )

    return G