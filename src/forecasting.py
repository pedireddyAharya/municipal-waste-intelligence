
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_forecasting_data(monthly, lags=3):
    """Create historical features for predicting monthly waste."""

    all_data = []

    for ward, group in monthly.groupby("ward"):
        group = group.sort_values("month").copy()

        # Ensure each ward has a complete monthly calendar.
        calendar = pd.date_range(
            group["month"].min(),
            group["month"].max(),
            freq="MS"
        )

        group = (
            group.set_index("month")
            .reindex(calendar)
            .rename_axis("month")
            .reset_index()
        )

        group["ward"] = ward

        # Previous months' waste quantities.
        for lag in range(1, lags + 1):
            group[f"lag_{lag}"] = (
                group["waste_kg"].shift(lag)
            )

        # Month number captures a basic seasonal pattern.
        group["month_num"] = group["month"].dt.month

        all_data.append(group)

    data = pd.concat(all_data, ignore_index=True)

    feature_columns = [
        *[f"lag_{i}" for i in range(1, lags + 1)],
        "month_num"
    ]

    data = data.dropna(
        subset=feature_columns + ["waste_kg"]
    )

    return data, feature_columns


def train_and_evaluate(monthly):
    """Train on earlier months and evaluate on later months."""

    data, feature_columns = prepare_forecasting_data(
        monthly,
        lags=3
    )

    if len(data) < 20:
        raise ValueError(
            "Not enough data to evaluate forecasting."
        )

    # Split chronologically to avoid training on future observations.
    data = data.sort_values("month")

    months = sorted(data["month"].unique())
    split_index = int(len(months) * 0.8)

    split_index = max(1, min(split_index, len(months) - 1))
    cutoff_month = months[split_index]

    train = data[data["month"] < cutoff_month].copy()
    test = data[data["month"] >= cutoff_month].copy()

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train[feature_columns],
        train["waste_kg"]
    )

    predictions = model.predict(
        test[feature_columns]
    )

    mae = mean_absolute_error(
        test["waste_kg"],
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            test["waste_kg"],
            predictions
        )
    )

    results = test[
        ["month", "ward", "waste_kg"]
    ].copy()

    results["predicted_waste_kg"] = predictions

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "training_records": len(train),
        "testing_records": len(test)
    }

    return model, metrics, results
