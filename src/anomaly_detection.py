
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df, contamination=0.015):
    """Flag unusual waste records for human review."""

    result = df.copy()

    features = ["waste_kg"]

    if "collection_trips" in result.columns:
        features.append("collection_trips")

    X = result[features].copy()

    X = X.replace(
        [float("inf"), float("-inf")],
        float("nan")
    )

    # Fill missing feature values with their median.
    X = X.fillna(X.median()).fillna(0)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )

    labels = model.fit_predict(X)

    result["anomaly_flag"] = labels == -1

    # Higher scores indicate more unusual observations.
    result["anomaly_score"] = (
        -model.decision_function(X)
    )

    return result.sort_values(
        "anomaly_score",
        ascending=False
    )
