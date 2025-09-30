"""Domain adversarial training for site removal in multi-omics integration"""
from typing import Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class DomainAdversarialIntegration:
    """
    Domain adversarial training to remove site effects from latent representations

    Uses adversarial learning to ensure latent factors are site-invariant while
    preserving biological signal.
    """

    def __init__(
        self,
        n_factors: int = 10,
        adversarial_weight: float = 0.1,
        n_iterations: int = 100,
        learning_rate: float = 0.01,
    ):
        """
        Initialize domain adversarial integration

        Args:
            n_factors: Number of latent factors
            adversarial_weight: Weight for adversarial loss (higher = more site removal)
            n_iterations: Number of training iterations
            learning_rate: Learning rate for optimization
        """
        self.n_factors = n_factors
        self.adversarial_weight = adversarial_weight
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

        # Model components
        self.encoder_weights: Optional[Dict[str, np.ndarray]] = None
        self.site_classifier: Optional[LogisticRegression] = None
        self.label_encoder: Optional[LabelEncoder] = None

    def fit_transform(
        self,
        data_dict: Dict[str, pd.DataFrame],
        site_labels: pd.Series,
        biological_labels: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Fit domain adversarial model and transform data

        Args:
            data_dict: Dictionary of omics data matrices
            site_labels: Site labels for each sample
            biological_labels: Optional biological group labels to preserve

        Returns:
            Site-invariant latent factors
        """
        # Prepare data
        X_concat, sample_ids = self._prepare_data(data_dict)

        # Align site labels
        site_labels = site_labels.loc[sample_ids]

        # Encode site labels
        self.label_encoder = LabelEncoder()
        site_encoded = self.label_encoder.fit_transform(site_labels)

        # Initialize encoder (simple linear transformation for now)
        n_features = X_concat.shape[1]
        self.encoder_weights = {}

        # Initialize with random weights
        W = np.random.randn(n_features, self.n_factors) * 0.01
        b = np.zeros(self.n_factors)

        # Training loop
        for iteration in range(self.n_iterations):
            # Forward pass: encode to latent space
            Z = X_concat @ W + b

            # Train site classifier on latent factors
            if iteration % 10 == 0:  # Update classifier every 10 iterations
                self.site_classifier = LogisticRegression(
                    max_iter=100, random_state=42
                )
                self.site_classifier.fit(Z, site_encoded)

            # Get classifier predictions
            site_pred = self.site_classifier.predict_proba(Z)
            site_true = np.eye(len(self.label_encoder.classes_))[site_encoded]

            # Adversarial loss gradient through latent factors
            # d_L_adv / d_Z = (site_pred - site_true) @ classifier_weights.T (simplified)
            # For gradient reversal, we negate this
            adversarial_grad_z = -self.adversarial_weight * (site_pred - site_true)

            # Reconstruction loss: maintain biological signal
            # L_recon = ||X - Z @ W^T||^2
            # d_L_recon / d_W = -2 * (X - Z @ W^T)^T @ Z / n = -2 * recon_error^T @ Z / n
            reconstruction = Z @ W.T
            recon_error = X_concat - reconstruction
            recon_grad_w = -2 * recon_error.T @ Z / len(X_concat)

            # Adversarial gradient w.r.t W: d_L_adv / d_W = X^T @ d_L_adv / d_Z / n
            # adversarial_grad_z is (n_samples, n_classes)
            # site_classifier.coef_ is (n_classes, n_factors)
            # Result should be (n_features, n_factors)
            adv_grad_w = X_concat.T @ (adversarial_grad_z @ self.site_classifier.coef_) / len(X_concat)

            # Update encoder weights (gradient descent for both)
            W = W - self.learning_rate * (recon_grad_w + adv_grad_w)

            # Monitor convergence (simplified)
            if iteration % 20 == 0:
                site_accuracy = (
                    self.site_classifier.predict(Z) == site_encoded
                ).mean()
                recon_error = np.mean((X_concat - reconstruction) ** 2)

                if iteration % 100 == 0:
                    print(
                        f"Iteration {iteration}: Site accuracy = {site_accuracy:.3f}, "
                        f"Recon error = {recon_error:.3f}"
                    )

        # Final encoding
        Z_final = X_concat @ W + b

        self.encoder_weights = {"W": W, "b": b}

        return pd.DataFrame(
            Z_final,
            index=sample_ids,
            columns=[f"Factor{i+1}" for i in range(self.n_factors)],
        )

    def _prepare_data(
        self, data_dict: Dict[str, pd.DataFrame]
    ) -> Tuple[np.ndarray, pd.Index]:
        """
        Prepare and concatenate multi-omics data

        Args:
            data_dict: Dictionary of data matrices

        Returns:
            Tuple of (concatenated data, sample IDs)
        """
        # Get union of all samples
        all_samples = set()
        for df in data_dict.values():
            all_samples.update(df.index)
        sample_ids = pd.Index(sorted(list(all_samples)))

        # Concatenate and align data
        data_list = []
        for view, df in data_dict.items():
            # Align to common sample set
            aligned = df.reindex(sample_ids, fill_value=0)

            # Standardize
            scaler = StandardScaler()
            scaled = scaler.fit_transform(aligned.fillna(0))

            data_list.append(scaled)

        X_concat = np.hstack(data_list)

        return X_concat, pd.Index(sample_ids)

    def transform(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Transform new data using fitted encoder

        Args:
            data_dict: Dictionary of data matrices

        Returns:
            Latent factors
        """
        if self.encoder_weights is None:
            raise ValueError("Model not fitted. Call fit_transform() first.")

        X_concat, sample_ids = self._prepare_data(data_dict)

        W = self.encoder_weights["W"]
        b = self.encoder_weights["b"]

        Z = X_concat @ W + b

        return pd.DataFrame(
            Z,
            index=sample_ids,
            columns=[f"Factor{i+1}" for i in range(self.n_factors)],
        )


def apply_adversarial_correction(
    latent_factors: pd.DataFrame,
    site_labels: pd.Series,
    correction_strength: float = 0.5,
) -> pd.DataFrame:
    """
    Apply adversarial correction to pre-computed latent factors

    This is a simpler post-hoc correction method that can be applied to
    factors from any integration method (e.g., MOFA).

    Args:
        latent_factors: Pre-computed latent factors (samples × factors)
        site_labels: Site labels for each sample
        correction_strength: Strength of correction (0-1, higher = more correction)

    Returns:
        Corrected latent factors
    """
    # Align site labels
    site_labels = site_labels.loc[latent_factors.index]

    # Encode sites
    label_encoder = LabelEncoder()
    site_encoded = label_encoder.fit_transform(site_labels)
    n_sites = len(label_encoder.classes_)

    if n_sites <= 1:
        return latent_factors

    # For each factor, remove site-related variance
    Z_corrected = latent_factors.copy()

    for factor_col in latent_factors.columns:
        z = latent_factors[factor_col].values

        # Fit linear model: z ~ site
        # Create site design matrix
        site_design = np.eye(n_sites)[site_encoded]

        # Fit: z = site_design @ beta + residual
        beta = np.linalg.lstsq(site_design, z, rcond=None)[0]

        # Remove site effect
        site_effect = site_design @ beta
        z_corrected = z - correction_strength * (site_effect - site_effect.mean())

        Z_corrected[factor_col] = z_corrected

    return Z_corrected


def evaluate_site_confounding(
    latent_factors: pd.DataFrame, site_labels: pd.Series
) -> Dict[str, float]:
    """
    Evaluate how much site information is retained in latent factors

    Args:
        latent_factors: Latent factors
        site_labels: Site labels

    Returns:
        Dictionary with evaluation metrics
    """
    site_labels = site_labels.loc[latent_factors.index]

    # Encode sites
    label_encoder = LabelEncoder()
    site_encoded = label_encoder.fit_transform(site_labels)

    # Train classifier to predict site from factors
    classifier = MLPClassifier(
        hidden_layer_sizes=(50,), max_iter=500, random_state=42
    )

    # Cross-validation accuracy
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(
        classifier, latent_factors.values, site_encoded, cv=5, scoring="accuracy"
    )

    # Random baseline
    random_accuracy = 1.0 / len(label_encoder.classes_)

    metrics = {
        "site_classification_accuracy": scores.mean(),
        "site_classification_std": scores.std(),
        "random_baseline": random_accuracy,
        "excess_site_information": max(0, scores.mean() - random_accuracy),
    }

    return metrics