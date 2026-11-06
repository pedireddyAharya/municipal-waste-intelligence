
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


FEATURES = [
    "avg_monthly_waste",
    "waste_variability",
    "trend_slope"
]


def prepare_features(ward_features):
    """Scale numerical features before clustering."""

    X = ward_features[FEATURES].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    valid = X.notna().all(axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.loc[valid])

    return X_scaled, valid, scaler


def run_kmeans(ward_features, n_clusters=3):
    """Cluster wards using K-Means."""

    result = ward_features.copy()
    result["kmeans_cluster"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )

    X_scaled, valid, scaler = prepare_features(
        result
    )

    if valid.sum() < 2:
        raise ValueError(
            "At least two valid wards are required."
        )

    k = min(n_clusters, int(valid.sum()))

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    result.loc[valid, "kmeans_cluster"] = labels

    score = None

    if 1 < len(set(labels)) < len(labels):
        score = silhouette_score(X_scaled, labels)

    return result, score, model


def run_dbscan(ward_features, eps=1.2, min_samples=3):
    """Cluster wards using DBSCAN."""

    result = ward_features.copy()
    result["dbscan_cluster"] = pd.Series(
        pd.NA, index=result.index, dtype="Int64"
    )

    X_scaled, valid, scaler = prepare_features(
        result
    )

    if valid.sum() < 2:
        raise ValueError(
            "At least two valid wards are required."
        )

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    labels = model.fit_predict(X_scaled)

    result.loc[valid, "dbscan_cluster"] = labels

    return result
