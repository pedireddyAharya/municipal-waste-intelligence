
from src.preprocessing import (
    load_and_clean_data,
    build_ward_features
)

from src.clustering import (
    run_kmeans,
    run_dbscan
)

file_path = (
    "data/municipal_waste_synthetic_ward_data.csv"
)

df = load_and_clean_data(file_path)

monthly, ward_features = build_ward_features(df)

# K-Means clustering
kmeans_results, score, model = run_kmeans(
    ward_features,
    n_clusters=3
)

print("\nK-MEANS CLUSTERING RESULTS")
print(
    kmeans_results[
        [
            "ward",
            "avg_monthly_waste",
            "trend_slope",
            "kmeans_cluster"
        ]
    ].to_string(index=False)
)

if score is not None:
    print("\nSilhouette Score:", round(score, 3))
else:
    print("\nSilhouette Score unavailable.")

# DBSCAN clustering
dbscan_results = run_dbscan(
    kmeans_results,
    eps=1.2,
    min_samples=3
)

print("\nDBSCAN CLUSTERING RESULTS")
print(
    dbscan_results[
        ["ward", "kmeans_cluster", "dbscan_cluster"]
    ].to_string(index=False)
)

print("\nK-Means cluster sizes:")
print(kmeans_results["kmeans_cluster"].value_counts())

print("\nDBSCAN cluster sizes:")
print(dbscan_results["dbscan_cluster"].value_counts())
