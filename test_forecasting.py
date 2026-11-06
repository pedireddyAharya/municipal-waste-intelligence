
from src.preprocessing import (
    load_and_clean_data,
    build_ward_features
)

from src.forecasting import train_and_evaluate

file_path = (
    "data/municipal_waste_synthetic_ward_data.csv"
)

df = load_and_clean_data(file_path)

monthly, ward_features = build_ward_features(df)

model, metrics, results = train_and_evaluate(monthly)

print("\nFORECASTING MODEL EVALUATION")

for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.2f}")
    else:
        print(f"{name}: {value}")

print("\nACTUAL VS PREDICTED WASTE")

print(results.head(15).to_string(index=False))

results.to_csv(
    "forecast_results.csv",
    index=False
)

print("\nResults saved to forecast_results.csv")
