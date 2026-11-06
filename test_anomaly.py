
from src.preprocessing import load_and_clean_data
from src.anomaly_detection import detect_anomalies

file_path = (
    "data/municipal_waste_synthetic_ward_data.csv"
)

df = load_and_clean_data(file_path)

results = detect_anomalies(
    df,
    contamination=0.015
)

flagged = results[results["anomaly_flag"]]

print("\nANOMALY DETECTION RESULTS")

print("Total records:", len(results))
print("Flagged records:", len(flagged))

print("\nMOST UNUSUAL RECORDS")

print(
    flagged[
        [
            "date",
            "ward",
            "waste_kg",
            "anomaly_score"
        ]
    ].head(20).to_string(index=False)
)

results.to_csv(
    "anomaly_results.csv",
    index=False
)

print("\nSaved results to anomaly_results.csv")
