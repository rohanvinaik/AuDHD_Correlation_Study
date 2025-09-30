"""Visualization tools for topology analysis results"""
from typing import Optional, Tuple, List
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

from .topology import TopologyAnalysisResult, PersistenceResult


def plot_topology_summary(
    result: TopologyAnalysisResult,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create comprehensive topology analysis summary plot

    Args:
        result: TopologyAnalysisResult
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Gap scores summary
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_gap_scores(ax1, result.gap_scores)

    # 2. MST edge length distribution
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_mst_distribution(ax2, result.mst_analysis)

    # 3. k-NN connectivity
    ax3 = fig.add_subplot(gs[0, 2])
    _plot_knn_connectivity(ax3, result.knn_connectivity)

    # 4. Spectral gaps
    ax4 = fig.add_subplot(gs[1, :])
    _plot_spectral_gaps(ax4, result.spectral_gaps)

    # 5. Persistence diagrams (if available)
    if result.persistence is not None:
        ax5 = fig.add_subplot(gs[2, 0])
        ax6 = fig.add_subplot(gs[2, 1])
        _plot_persistence_diagrams(ax5, result.persistence, dimension=0)
        _plot_persistence_diagrams(ax6, result.persistence, dimension=1)

        # Persistence entropy
        ax7 = fig.add_subplot(gs[2, 2])
        _plot_persistence_entropy(ax7, result.persistence)
    else:
        # Hypothesis test summary
        ax5 = fig.add_subplot(gs[2, :])
        _plot_hypothesis_summary(ax5, result)

    # Overall title
    interpretation = result.overall_interpretation or "unknown"
    fig.suptitle(
        f"Topology Analysis: {interpretation.upper()}",
        fontsize=16,
        fontweight='bold',
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def _plot_gap_scores(ax: plt.Axes, gap_scores) -> None:
    """Plot gap scores"""
    scores = {
        'Gap\nScore': gap_scores.score,
        'Within\nDist': gap_scores.within_cluster_distance / 10,  # Normalize
        'Between\nDist': gap_scores.between_cluster_distance / 10,
    }

    colors = ['#2ecc71' if gap_scores.score > 0.5 else '#e74c3c']
    bars = ax.bar(range(len(scores)), scores.values(), color=colors + ['#3498db', '#e67e22'])

    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(scores.keys(), fontsize=9)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score')
    ax.set_title('Density Gap Scores')

    # Add interpretation
    interp_color = {'separated': '#2ecc71', 'spectrum': '#e74c3c', 'intermediate': '#f39c12'}
    ax.text(
        0.5, 0.95, f"Interpretation: {gap_scores.interpretation}",
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor=interp_color.get(gap_scores.interpretation, 'gray'), alpha=0.3),
    )


def _plot_mst_distribution(ax: plt.Axes, mst_analysis: dict) -> None:
    """Plot MST edge statistics"""
    stats = mst_analysis['gap_statistics']

    categories = ['Within\nCluster', 'Between\nCluster']
    values = [
        stats.get('within_cluster_mean', stats['mean_within_distance']),
        stats.get('between_cluster_mean', stats['mean_between_distance']),
    ]

    colors = ['#3498db', '#e74c3c']
    ax.bar(range(len(categories)), values, color=colors)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel('Mean Edge Length')
    ax.set_title('MST Edge Statistics')

    # Add gap ratio
    gap_ratio = stats.get('cluster_gap_ratio', stats['gap_ratio'])
    ax.text(
        0.5, 0.95, f"Gap Ratio: {gap_ratio:.2f}",
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3),
    )


def _plot_knn_connectivity(ax: plt.Axes, knn_test: dict) -> None:
    """Plot k-NN connectivity metrics"""
    metrics = {
        'Edge\nPurity': knn_test['edge_purity'],
        'Modularity': min(1.0, knn_test['modularity']),
    }

    colors = ['#2ecc71' if knn_test['edge_purity'] > 0.7 else '#e74c3c',
              '#2ecc71' if knn_test['modularity'] > 0.3 else '#e74c3c']

    ax.bar(range(len(metrics)), metrics.values(), color=colors)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics.keys(), fontsize=9)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score')
    ax.set_title('k-NN Graph Connectivity')

    # Add interpretation
    interp_color = {'separated': '#2ecc71', 'spectrum': '#e74c3c', 'intermediate': '#f39c12'}
    ax.text(
        0.5, 0.95, f"Interpretation: {knn_test['interpretation']}",
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor=interp_color.get(knn_test['interpretation'], 'gray'), alpha=0.3),
    )


def _plot_spectral_gaps(ax: plt.Axes, spectral_analysis: dict) -> None:
    """Plot spectral gaps"""
    gaps = spectral_analysis['gaps']['all_gaps']

    # Plot all gaps
    ax.plot(range(1, len(gaps) + 1), gaps, 'o-', color='#3498db', linewidth=2, markersize=6)

    # Highlight largest gap
    largest_idx = spectral_analysis['gaps']['largest_gap_index']
    ax.plot(largest_idx + 1, gaps[largest_idx], 'o', color='#e74c3c', markersize=12,
            label=f'Largest gap (k={largest_idx + 1})')

    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Spectral Gap')
    ax.set_title('Spectral Gaps in Graph Laplacian')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add interpretation
    test = spectral_analysis['test']
    interp_color = {'separated': '#2ecc71', 'spectrum': '#e74c3c', 'intermediate': '#f39c12'}
    ax.text(
        0.98, 0.95,
        f"Interpretation: {test['interpretation']}\n"
        f"Confidence: {test['confidence']:.2f}\n"
        f"Est. clusters: {test['n_clusters_estimate']}",
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor=interp_color.get(test['interpretation'], 'gray'), alpha=0.3),
    )


def _plot_persistence_diagrams(
    ax: plt.Axes,
    persistence: PersistenceResult,
    dimension: int = 0,
) -> None:
    """Plot persistence diagram for specific dimension"""
    if dimension >= len(persistence.diagrams):
        ax.text(0.5, 0.5, f'No H{dimension} features', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'H{dimension} Persistence Diagram')
        return

    dgm = persistence.diagrams[dimension]

    # Remove infinite points for visualization
    finite_dgm = dgm[dgm[:, 1] < np.inf]

    if len(finite_dgm) == 0:
        ax.text(0.5, 0.5, f'No finite H{dimension} features', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'H{dimension} Persistence Diagram')
        return

    # Plot birth-death pairs
    births = finite_dgm[:, 0]
    deaths = finite_dgm[:, 1]
    lifetimes = deaths - births

    # Color by lifetime
    scatter = ax.scatter(births, deaths, c=lifetimes, cmap='viridis', s=50, alpha=0.6)

    # Diagonal line
    max_val = max(np.max(births), np.max(deaths))
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Birth = Death')

    # Color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Persistence')

    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(f'H{dimension} Persistence Diagram ({len(finite_dgm)} features)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')


def _plot_persistence_entropy(ax: plt.Axes, persistence: PersistenceResult) -> None:
    """Plot persistence entropy"""
    if persistence.persistence_entropy is None:
        return

    dimensions = sorted(persistence.persistence_entropy.keys())
    entropies = [persistence.persistence_entropy[d] for d in dimensions]

    colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(dimensions)]
    ax.bar(range(len(dimensions)), entropies, color=colors)

    ax.set_xticks(range(len(dimensions)))
    ax.set_xticklabels([f'H{d}' for d in dimensions])
    ax.set_ylabel('Entropy')
    ax.set_title('Persistence Entropy by Dimension')
    ax.grid(True, alpha=0.3, axis='y')

    # Add interpretation
    if persistence.interpretation:
        interp_color = {'separated': '#2ecc71', 'spectrum': '#e74c3c', 'intermediate': '#f39c12'}
        ax.text(
            0.5, 0.95, f"Topology: {persistence.interpretation}",
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor=interp_color.get(persistence.interpretation, 'gray'), alpha=0.3),
        )


def _plot_hypothesis_summary(ax: plt.Axes, result: TopologyAnalysisResult) -> None:
    """Plot hypothesis test summary"""
    if result.hypothesis_test is None:
        return

    test = result.hypothesis_test

    # Overall separation score
    score = test['separation_score']
    confidence = test['confidence']

    # Create score visualization
    ax.barh([0], [score], color='#2ecc71' if score > 0.6 else '#e74c3c' if score < 0.4 else '#f39c12')
    ax.barh([1], [confidence], color='#3498db')

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Separation\nScore', 'Confidence'])
    ax.set_xlim([0, 1])
    ax.set_xlabel('Score')
    ax.set_title('Integrated Hypothesis Test')

    # Add vertical lines for thresholds
    ax.axvline(0.4, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.6, color='gray', linestyle='--', alpha=0.5)

    # Add interpretation
    interpretation = result.overall_interpretation or "unknown"
    interp_color = {'separated': '#2ecc71', 'spectrum': '#e74c3c', 'intermediate': '#f39c12'}
    ax.text(
        0.98, 0.95,
        f"Overall: {interpretation.upper()}\n"
        f"Based on {test['n_tests']} tests",
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=11,
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=interp_color.get(interpretation, 'gray'), alpha=0.5),
    )


def plot_mst_with_gaps(
    X: np.ndarray,
    labels: np.ndarray,
    mst_analyzer,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize minimum spanning tree with gap edges highlighted

    Args:
        X: Data matrix (for 2D visualization, will use first 2 dims)
        labels: Cluster labels
        mst_analyzer: Fitted MinimumSpanningTreeAnalyzer
        figsize: Figure size
        save_path: Optional save path

    Returns:
        matplotlib Figure
    """
    if X.shape[1] > 2:
        # Use first 2 dimensions or apply PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
    else:
        X_2d = X

    fig, ax = plt.subplots(figsize=figsize)

    # Plot edges
    mst_array = mst_analyzer.mst_.toarray()

    for i in range(mst_array.shape[0]):
        for j in range(i + 1, mst_array.shape[1]):
            weight = max(mst_array[i, j], mst_array[j, i])
            if weight > 0:
                # Color by whether it's a gap edge
                if labels[i] != labels[j]:
                    color = '#e74c3c'
                    linewidth = 2
                    alpha = 0.8
                else:
                    color = '#95a5a6'
                    linewidth = 1
                    alpha = 0.3

                ax.plot([X_2d[i, 0], X_2d[j, 0]], [X_2d[i, 1], X_2d[j, 1]],
                        color=color, linewidth=linewidth, alpha=alpha, zorder=1)

    # Plot points
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        if label == -1:
            continue
        mask = labels == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]], label=f'Cluster {label}',
                   s=50, zorder=2, edgecolors='black', linewidths=0.5)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Minimum Spanning Tree with Cluster Boundaries')
    ax.legend()

    # Add legend for edges
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#95a5a6', linewidth=1, alpha=0.3, label='Within-cluster'),
        Line2D([0], [0], color='#e74c3c', linewidth=2, alpha=0.8, label='Between-cluster'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_knn_graph(
    X: np.ndarray,
    labels: np.ndarray,
    knn_analyzer,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize k-NN graph with cluster assignments

    Args:
        X: Data matrix (for 2D visualization)
        labels: Cluster labels
        knn_analyzer: Fitted KNNGraphConnectivityAnalyzer
        figsize: Figure size
        save_path: Optional save path

    Returns:
        matplotlib Figure
    """
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
    else:
        X_2d = X

    fig, ax = plt.subplots(figsize=figsize)

    # Plot k-NN edges
    graph_coo = knn_analyzer.graph_.tocoo()

    for i, j, v in zip(graph_coo.row, graph_coo.col, graph_coo.data):
        # Color by whether edge crosses cluster boundary
        if labels[i] != labels[j]:
            color = '#e74c3c'
            alpha = 0.5
            linewidth = 1.5
        else:
            color = '#95a5a6'
            alpha = 0.1
            linewidth = 0.5

        ax.plot([X_2d[i, 0], X_2d[j, 0]], [X_2d[i, 1], X_2d[j, 1]],
                color=color, alpha=alpha, linewidth=linewidth, zorder=1)

    # Plot points
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        if label == -1:
            continue
        mask = labels == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[colors[i]], label=f'Cluster {label}',
                   s=50, zorder=2, edgecolors='black', linewidths=0.5)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(f'k-NN Graph (k={knn_analyzer.n_neighbors})')

    # Add statistics
    if knn_analyzer.connectivity_:
        edge_purity = knn_analyzer.connectivity_.get('edge_purity', 0)
        ax.text(
            0.02, 0.98, f"Edge Purity: {edge_purity:.3f}",
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
        )

    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_density_landscape(
    X: np.ndarray,
    labels: np.ndarray,
    density_analyzer,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize density landscape with clusters

    Args:
        X: Data matrix
        labels: Cluster labels
        density_analyzer: Fitted DensityGapAnalyzer
        figsize: Figure size
        save_path: Optional save path

    Returns:
        matplotlib Figure
    """
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
    else:
        X_2d = X

    fig, ax = plt.subplots(figsize=figsize)

    # Plot density as background
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=density_analyzer.densities_,
        cmap='YlOrRd',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidths=0.5,
    )

    # Overlay cluster labels
    unique_labels = np.unique(labels[labels >= 0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   facecolors='none', edgecolors=[colors[i]],
                   s=150, linewidths=2, label=f'Cluster {label}')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Local Density')

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Density Landscape')
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_topology_report(
    result: TopologyAnalysisResult,
    output_path: str,
) -> None:
    """
    Create comprehensive HTML report for topology analysis

    Args:
        result: TopologyAnalysisResult
        output_path: Path to save HTML report
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Topology Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 30px;
            }}
            .section {{
                background-color: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric {{
                display: inline-block;
                margin: 10px;
                padding: 15px;
                background-color: #ecf0f1;
                border-radius: 5px;
                min-width: 150px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .metric-label {{
                font-size: 12px;
                color: #7f8c8d;
            }}
            .interpretation {{
                padding: 15px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 18px;
                text-align: center;
                margin: 20px 0;
            }}
            .separated {{ background-color: #2ecc71; color: white; }}
            .spectrum {{ background-color: #e74c3c; color: white; }}
            .intermediate {{ background-color: #f39c12; color: white; }}
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #34495e;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Topology Analysis Report</h1>
            <p>Testing Separation vs. Spectrum Hypothesis</p>
        </div>

        <div class="section">
            <h2>Overall Interpretation</h2>
            <div class="interpretation {result.overall_interpretation}">
                {result.overall_interpretation.upper() if result.overall_interpretation else "UNKNOWN"}
            </div>
        </div>

        <div class="section">
            <h2>Integrated Hypothesis Test</h2>
            <div class="metric">
                <div class="metric-value">{result.hypothesis_test['separation_score']:.3f}</div>
                <div class="metric-label">Separation Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result.hypothesis_test['confidence']:.3f}</div>
                <div class="metric-label">Confidence</div>
            </div>
            <div class="metric">
                <div class="metric-value">{result.hypothesis_test['n_tests']}</div>
                <div class="metric-label">Number of Tests</div>
            </div>
        </div>

        <div class="section">
            <h2>Density Gap Analysis</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Gap Score</td>
                    <td>{result.gap_scores.score:.3f}</td>
                </tr>
                <tr>
                    <td>Within-Cluster Distance</td>
                    <td>{result.gap_scores.within_cluster_distance:.3f}</td>
                </tr>
                <tr>
                    <td>Between-Cluster Distance</td>
                    <td>{result.gap_scores.between_cluster_distance:.3f}</td>
                </tr>
                <tr>
                    <td>Gap Statistic</td>
                    <td>{result.gap_scores.gap_statistic:.3f}</td>
                </tr>
                <tr>
                    <td>P-value</td>
                    <td>{result.gap_scores.p_value:.4f}</td>
                </tr>
                <tr>
                    <td>Interpretation</td>
                    <td><strong>{result.gap_scores.interpretation}</strong></td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>k-NN Graph Connectivity</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Edge Purity</td>
                    <td>{result.knn_connectivity['edge_purity']:.3f}</td>
                </tr>
                <tr>
                    <td>Modularity</td>
                    <td>{result.knn_connectivity['modularity']:.3f}</td>
                </tr>
                <tr>
                    <td>Number of Components</td>
                    <td>{result.knn_connectivity['n_components']}</td>
                </tr>
                <tr>
                    <td>Interpretation</td>
                    <td><strong>{result.knn_connectivity['interpretation']}</strong></td>
                </tr>
            </table>
        </div>

        <div class="section">
            <h2>Spectral Gap Analysis</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Estimated Clusters</td>
                    <td>{result.spectral_gaps['test']['n_clusters_estimate']}</td>
                </tr>
                <tr>
                    <td>Eigengap Ratio</td>
                    <td>{result.spectral_gaps['test']['eigengap_ratio']:.3f}</td>
                </tr>
                <tr>
                    <td>Significant Gaps</td>
                    <td>{result.spectral_gaps['test']['n_significant_gaps']}</td>
                </tr>
                <tr>
                    <td>Confidence</td>
                    <td>{result.spectral_gaps['test']['confidence']:.3f}</td>
                </tr>
                <tr>
                    <td>Interpretation</td>
                    <td><strong>{result.spectral_gaps['test']['interpretation']}</strong></td>
                </tr>
            </table>
        </div>

        {_persistence_section_html(result.persistence) if result.persistence else ''}

        <div class="section">
            <h2>Interpretation Guide</h2>
            <ul>
                <li><strong>Separated:</strong> Data exhibits discrete clusters with clear boundaries. High separation score (>0.6), significant gaps in MST, high edge purity, strong spectral gaps.</li>
                <li><strong>Spectrum:</strong> Data exhibits continuous variation without clear boundaries. Low separation score (<0.4), no significant gaps, low edge purity, weak spectral gaps.</li>
                <li><strong>Intermediate:</strong> Data shows mixed characteristics, potentially hierarchical structure or overlapping clusters.</li>
            </ul>
        </div>
    </body>
    </html>
    """

    with open(output_path, 'w') as f:
        f.write(html)


def _persistence_section_html(persistence: Optional[PersistenceResult]) -> str:
    """Generate HTML section for persistence results"""
    if persistence is None:
        return ''

    entropy_rows = ''
    if persistence.persistence_entropy:
        for dim, entropy in persistence.persistence_entropy.items():
            entropy_rows += f"""
            <tr>
                <td>H{dim} Entropy</td>
                <td>{entropy:.3f}</td>
            </tr>
            """

    return f"""
    <div class="section">
        <h2>Persistent Homology</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            {entropy_rows}
            <tr>
                <td>Interpretation</td>
                <td><strong>{persistence.interpretation if persistence.interpretation else 'N/A'}</strong></td>
            </tr>
        </table>
    </div>
    """