"""Random Forest classifier for cluster prediction

Trains classifiers to predict cluster membership and extract feature importance.
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


@dataclass
class ClusterClassifierResult:
    """Result of cluster classification"""
    classifier: RandomForestClassifier
    feature_names: List[str]
    cluster_labels: np.ndarray

    # Performance metrics
    train_accuracy: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    classification_report: Dict
    confusion_matrix: np.ndarray

    # Feature importance
    feature_importance: Dict[str, float]
    feature_importance_std: Dict[str, float]

    # Per-cluster metrics
    per_cluster_precision: Dict[int, float]
    per_cluster_recall: Dict[int, float]
    per_cluster_f1: Dict[int, float]


def train_cluster_classifier(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    max_features: str = 'sqrt',
    class_weight: str = 'balanced',
    n_jobs: int = -1,
    random_state: int = 42,
    cv_folds: int = 5,
) -> ClusterClassifierResult:
    """
    Train Random Forest classifier for cluster prediction

    Args:
        X: Feature matrix (n_samples, n_features)
        labels: Cluster labels (n_samples,)
        feature_names: Feature names (if None, use indices)
        n_estimators: Number of trees
        max_depth: Maximum tree depth (None = unlimited)
        min_samples_split: Minimum samples to split
        min_samples_leaf: Minimum samples per leaf
        max_features: Number of features for best split
        class_weight: Class weights ('balanced' or None)
        n_jobs: Number of parallel jobs
        random_state: Random seed
        cv_folds: Number of cross-validation folds

    Returns:
        ClusterClassifierResult with classifier and metrics
    """
    # Filter out noise points (-1 labels)
    valid_mask = labels >= 0
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]

    if len(X_valid) == 0:
        raise ValueError("No valid cluster labels found")

    # Feature names
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    unique_clusters = np.unique(labels_valid)
    n_clusters = len(unique_clusters)

    if n_clusters < 2:
        raise ValueError("Need at least 2 clusters for classification")

    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
        oob_score=True,
    )

    clf.fit(X_valid, labels_valid)

    # Training accuracy
    train_accuracy = clf.score(X_valid, labels_valid)

    # Cross-validation
    try:
        cv_scores = cross_val_score(
            clf, X_valid, labels_valid,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
            n_jobs=n_jobs,
        )
        cv_accuracy_mean = np.mean(cv_scores)
        cv_accuracy_std = np.std(cv_scores)
    except Exception as e:
        warnings.warn(f"Cross-validation failed: {e}")
        cv_accuracy_mean = train_accuracy
        cv_accuracy_std = 0.0

    # Predictions for metrics
    y_pred = clf.predict(X_valid)

    # Classification report
    report = classification_report(labels_valid, y_pred, output_dict=True, zero_division=0)

    # Confusion matrix
    conf_matrix = confusion_matrix(labels_valid, y_pred)

    # Feature importance
    importances = clf.feature_importances_

    # Calculate feature importance std across trees
    tree_importances = np.array([tree.feature_importances_ for tree in clf.estimators_])
    importance_std = np.std(tree_importances, axis=0)

    feature_importance = {
        feature_names[i]: float(importances[i])
        for i in range(len(feature_names))
    }

    feature_importance_std = {
        feature_names[i]: float(importance_std[i])
        for i in range(len(feature_names))
    }

    # Per-cluster metrics
    per_cluster_precision = {}
    per_cluster_recall = {}
    per_cluster_f1 = {}

    for cluster_id in unique_clusters:
        cluster_str = str(cluster_id)
        if cluster_str in report:
            per_cluster_precision[int(cluster_id)] = report[cluster_str]['precision']
            per_cluster_recall[int(cluster_id)] = report[cluster_str]['recall']
            per_cluster_f1[int(cluster_id)] = report[cluster_str]['f1-score']

    return ClusterClassifierResult(
        classifier=clf,
        feature_names=feature_names,
        cluster_labels=labels_valid,
        train_accuracy=train_accuracy,
        cv_accuracy_mean=cv_accuracy_mean,
        cv_accuracy_std=cv_accuracy_std,
        classification_report=report,
        confusion_matrix=conf_matrix,
        feature_importance=feature_importance,
        feature_importance_std=feature_importance_std,
        per_cluster_precision=per_cluster_precision,
        per_cluster_recall=per_cluster_recall,
        per_cluster_f1=per_cluster_f1,
    )


def get_top_features(
    classifier_result: ClusterClassifierResult,
    n_features: int = 20,
) -> List[Tuple[str, float]]:
    """
    Get top features by importance

    Args:
        classifier_result: ClusterClassifierResult
        n_features: Number of top features to return

    Returns:
        List of (feature_name, importance) tuples, sorted by importance
    """
    sorted_features = sorted(
        classifier_result.feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_features[:n_features]


def predict_cluster(
    classifier_result: ClusterClassifierResult,
    X_new: np.ndarray,
    return_probabilities: bool = False,
) -> np.ndarray:
    """
    Predict cluster labels for new samples

    Args:
        classifier_result: ClusterClassifierResult
        X_new: New samples (n_samples, n_features)
        return_probabilities: Return class probabilities instead of labels

    Returns:
        Predicted labels or probabilities
    """
    if return_probabilities:
        return classifier_result.classifier.predict_proba(X_new)
    else:
        return classifier_result.classifier.predict(X_new)


def get_cluster_separability(
    classifier_result: ClusterClassifierResult,
) -> Dict[str, float]:
    """
    Compute cluster separability metrics

    Args:
        classifier_result: ClusterClassifierResult

    Returns:
        Dictionary with separability metrics
    """
    report = classifier_result.classification_report

    # Overall metrics
    accuracy = classifier_result.train_accuracy
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']

    # Separability score (0-1, higher = better separation)
    # Combines accuracy and macro F1
    separability = 0.5 * accuracy + 0.5 * macro_f1

    return {
        'accuracy': accuracy,
        'cv_accuracy': classifier_result.cv_accuracy_mean,
        'cv_accuracy_std': classifier_result.cv_accuracy_std,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'separability_score': separability,
        'oob_score': classifier_result.classifier.oob_score_,
    }


def analyze_misclassifications(
    classifier_result: ClusterClassifierResult,
    X: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Analyze misclassified samples

    Args:
        classifier_result: ClusterClassifierResult
        X: Feature matrix

    Returns:
        Dictionary with misclassification analysis
    """
    # Get predictions
    y_pred = classifier_result.classifier.predict(X)
    y_true = classifier_result.cluster_labels

    # Misclassified samples
    misclassified_mask = y_pred != y_true
    misclassified_indices = np.where(misclassified_mask)[0]

    # Get probabilities
    probabilities = classifier_result.classifier.predict_proba(X)

    # Confidence (max probability)
    confidence = np.max(probabilities, axis=1)

    # Low confidence threshold
    low_confidence_mask = confidence < 0.5

    return {
        'misclassified_indices': misclassified_indices,
        'misclassified_true_labels': y_true[misclassified_mask],
        'misclassified_pred_labels': y_pred[misclassified_mask],
        'misclassified_confidence': confidence[misclassified_mask],
        'low_confidence_indices': np.where(low_confidence_mask)[0],
        'mean_confidence': np.mean(confidence),
        'mean_confidence_correct': np.mean(confidence[~misclassified_mask]),
        'mean_confidence_incorrect': np.mean(confidence[misclassified_mask]) if np.any(misclassified_mask) else 0.0,
    }