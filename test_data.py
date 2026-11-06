from src.preprocessing import (
    load_and_clean_data,
    build_ward_features
)

file_path = (
    "data/municipal_waste_synthetic_ward_data.csv"
)

df = load_and_clean_data(file_path)

monthly, ward_features = build_ward_features(df)

print("\nCLEANED DATA")
print(df.head())

print("\nDATASET SUMMARY")
print("Total records:", len(df))
print("Number of wards:", df["ward"].nunique())

print("\nWARD FEATURES")
print(ward_features.head())

print("\nMONTHLY DATA")
print(monthly.head())