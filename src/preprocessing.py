
import pandas as pd
import numpy as np


def load_and_clean_data(file_path):
    """Load and validate municipal waste records."""

    df = pd.read_csv(file_path)

    required = {"date", "ward", "waste_kg"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )

    df["ward"] = (
        df["ward"].astype("string").str.strip()
    )

    df["waste_kg"] = pd.to_numeric(
        df["waste_kg"], errors="coerce"
    )

    # Remove records with invalid essential values.
    df = df.dropna(
        subset=["date", "ward", "waste_kg"]
    )

    df = df[
        (df["ward"] != "") &
        (df["waste_kg"] >= 0)
    ].copy()

    # Remove exact duplicate rows.
    df = df.drop_duplicates()

    # Create a month column for aggregation.
    df["month"] = (
        df["date"].dt.to_period("M").dt.to_timestamp()
    )

    return df.sort_values(
        ["ward", "date"]
    ).reset_index(drop=True)


def build_ward_features(df):
    """Create statistical features for each ward."""

    monthly = (
        df.groupby(["ward", "month"], as_index=False)
        .agg(waste_kg=("waste_kg", "sum"))
        .sort_values(["ward", "month"])
    )

    results = []

    for ward, group in monthly.groupby("ward"):
        values = group["waste_kg"].to_numpy()

        slope = (
            np.polyfit(
                np.arange(len(values)), values, 1
            )[0]
            if len(values) >= 2
            else 0.0
        )

        first = values[0]
        last = values[-1]

        growth = (
            ((last - first) / first) * 100
            if first > 0
            else np.nan
        )

        results.append({
            "ward": ward,
            "avg_monthly_waste": values.mean(),
            "waste_variability": values.std(),
            "trend_slope": slope,
            "growth_pct": growth,
            "observed_months": len(values)
        })

    return monthly, pd.DataFrame(results)
