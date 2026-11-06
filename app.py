
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

from src.preprocessing import (
    load_and_clean_data,
    build_ward_features
)
from src.clustering import run_kmeans, run_dbscan
from src.forecasting import train_and_evaluate
from src.anomaly_detection import detect_anomalies


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Smart Municipal Waste Management",
    page_icon="♻️",
    layout="wide"
)

st.title("Smart Municipal Waste Management")
st.write(
    "Monitor waste collection, understand ward-level patterns, "
    "estimate future waste, and identify records that need attention."
)

st.caption(
    "Academic prototype | Current dataset contains simulated data"
)

st.divider()


# ==========================================================
# 2. LOAD DATA
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_DATA = (
    PROJECT_DIR
    / "data"
    / "municipal_waste_synthetic_ward_data.csv"
)

st.sidebar.title("Waste Management")
st.sidebar.caption("Data and analysis settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload waste collection data",
    type=["csv"],
    help="Required columns: date, ward, waste_kg"
)

if uploaded_file is not None:
    data_source = uploaded_file
else:
    data_source = DEFAULT_DATA

try:
    df = load_and_clean_data(data_source)

except Exception as error:
    st.error(f"Unable to load the data: {error}")
    st.info(
        "Check that the CSV is available and contains "
        "date, ward, and waste_kg columns."
    )
    st.stop()

if df.empty:
    st.error("No valid records were found in the dataset.")
    st.stop()

if df["ward"].nunique() < 2:
    st.error("At least two wards are required for ward comparison.")
    st.stop()

is_synthetic = (
    "data_source" in df.columns
    and df["data_source"]
    .astype(str)
    .str.contains("synthetic", case=False)
    .any()
)

if is_synthetic:
    st.warning(
        "Demonstration mode: this dataset contains simulated data. "
        "The results do not represent verified municipal measurements."
    )
else:
    st.info(
        "Review the dataset source and data quality before using "
        "these results for municipal decisions."
    )


# ==========================================================
# 3. SIDEBAR SETTINGS
# ==========================================================

st.sidebar.subheader("Analysis Settings")

max_clusters = min(6, int(df["ward"].nunique()))

n_clusters = st.sidebar.slider(
    "Number of ward groups",
    min_value=2,
    max_value=max_clusters,
    value=min(3, max_clusters),
    help="Controls how many groups of similar wards are created."
)

contamination = st.sidebar.slider(
    "Sensitivity to unusual records",
    min_value=0.005,
    max_value=0.05,
    value=0.015,
    step=0.005,
    format="%.3f",
    help=(
        "Higher values flag more records for review. "
        "This is a model setting, not a measured error rate."
    )
)

st.sidebar.divider()

st.sidebar.caption(
    "Group analysis, forecasting and unusual-record detection "
    "are calculated from the selected dataset."
)


# ==========================================================
# 4. RUN ANALYSIS
# ==========================================================

with st.spinner("Analyzing waste collection records..."):

    monthly, ward_features = build_ward_features(df)

    clustered, silhouette, kmeans_model = run_kmeans(
        ward_features,
        n_clusters=n_clusters
    )

    clustered = run_dbscan(clustered)

    anomaly_results = detect_anomalies(
        df,
        contamination=contamination
    )

    flagged = anomaly_results[
        anomaly_results["anomaly_flag"]
    ].copy()


# ==========================================================
# 5. GIVE WARD GROUPS SIMPLE NAMES
# ==========================================================

group_summary = (
    clustered.groupby("kmeans_cluster", dropna=True)
    .agg(
        average_waste=("avg_monthly_waste", "mean"),
        average_trend=("trend_slope", "mean"),
        number_of_wards=("ward", "count")
    )
    .sort_values("average_waste")
)

group_count = len(group_summary)

group_levels = {
    2: ["Lower waste", "Higher waste"],
    3: ["Lower waste", "Moderate waste", "Higher waste"],
    4: [
        "Lower waste",
        "Below-average waste",
        "Above-average waste",
        "Higher waste"
    ],
    5: [
        "Lowest waste",
        "Lower waste",
        "Moderate waste",
        "Higher waste",
        "Highest waste"
    ],
    6: [
        "Lowest waste",
        "Low waste",
        "Moderate-low waste",
        "Moderate-high waste",
        "High waste",
        "Highest waste"
    ]
}

levels = group_levels.get(
    group_count,
    [f"Waste group {i + 1}" for i in range(group_count)]
)

group_names = {}

for position, cluster_id in enumerate(group_summary.index):

    average_waste = group_summary.loc[
        cluster_id, "average_waste"
    ]

    average_trend = group_summary.loc[
        cluster_id, "average_trend"
    ]

    threshold = max(1.0, abs(average_waste) * 0.01)

    if average_trend > threshold:
        trend_label = "increasing"
    elif average_trend < -threshold:
        trend_label = "decreasing"
    else:
        trend_label = "relatively stable"

    group_names[cluster_id] = (
        f"{levels[position]} · {trend_label}"
    )

clustered["ward_group"] = (
    clustered["kmeans_cluster"].map(group_names)
)


# ==========================================================
# 6. KEY DASHBOARD METRICS
# ==========================================================

total_waste = df["waste_kg"].sum()

ward_count = df["ward"].nunique()

rising_count = int(
    (clustered["trend_slope"] > 0).sum()
)

flagged_count = len(flagged)

first_date = df["date"].min().strftime("%d %b %Y")
last_date = df["date"].max().strftime("%d %b %Y")

st.subheader("Waste Management Overview")

st.caption(
    f"Data period: {first_date} to {last_date}"
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total waste recorded",
    f"{total_waste:,.0f} kg"
)

c2.metric(
    "Wards analyzed",
    f"{ward_count}"
)

c3.metric(
    "Wards with increasing trends",
    f"{rising_count}"
)

c4.metric(
    "Records needing review",
    f"{flagged_count:,}"
)

st.caption(
    "Increasing trends are based on fitted historical trends. "
    "They do not automatically indicate a sanitation problem."
)


# ==========================================================
# 7. USER-FRIENDLY NAVIGATION
# ==========================================================

overview_tab, wards_tab, forecast_tab, attention_tab = st.tabs([
    "Overview",
    "Ward Analysis",
    "Waste Forecast",
    "Attention Required"
])


# ==========================================================
# 8. OVERVIEW
# ==========================================================

with overview_tab:

    st.subheader("How is waste collection changing?")

    monthly_total = (
        monthly.groupby("month", as_index=False)
        .agg(waste_kg=("waste_kg", "sum"))
        .sort_values("month")
    )

    fig = px.line(
        monthly_total,
        x="month",
        y="waste_kg",
        markers=True,
        labels={
            "month": "Month",
            "waste_kg": "Total waste collected (kg)"
        }
    )

    fig.update_layout(
        legend_title_text="",
        margin=dict(l=10, r=10, t=25, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Explore an individual ward")

    selected_ward = st.selectbox(
        "Select a ward",
        sorted(df["ward"].unique()),
        key="overview_ward"
    )

    ward_history = monthly[
        monthly["ward"] == selected_ward
    ].sort_values("month")

    fig_ward = px.line(
        ward_history,
        x="month",
        y="waste_kg",
        markers=True,
        labels={
            "month": "Month",
            "waste_kg": "Waste collected (kg)"
        }
    )

    st.plotly_chart(
        fig_ward,
        use_container_width=True
    )

    selected_features = clustered[
        clustered["ward"] == selected_ward
    ]

    if not selected_features.empty:

        info = selected_features.iloc[0]

        group_name = info["ward_group"]

        st.markdown(f"**Ward group:** {group_name}")

        st.write(
            f"Average monthly waste: "
            f"{info['avg_monthly_waste']:,.0f} kg"
        )

        if info["trend_slope"] > 0:
            st.write(
                "The fitted historical trend is increasing. "
                "Review recent records to understand the change."
            )
        elif info["trend_slope"] < 0:
            st.write(
                "The fitted historical trend is decreasing. "
                "Check whether the pattern continues."
            )
        else:
            st.write(
                "The fitted historical trend is approximately stable."
            )


# ==========================================================
# 9. WARD ANALYSIS
# ==========================================================

with wards_tab:

    st.subheader("Understand the different types of wards")

    st.write(
        "Wards are grouped according to their average waste, "
        "variation in waste, and historical trends."
    )

    group_chart = px.scatter(
        clustered,
        x="avg_monthly_waste",
        y="trend_slope",
        color="ward_group",
        hover_name="ward",
        size="waste_variability",
        labels={
            "avg_monthly_waste": "Average monthly waste (kg)",
            "trend_slope": "Monthly trend (kg per observed month)",
            "ward_group": "Ward group",
            "waste_variability": "Waste variation"
        }
    )

    group_chart.update_layout(
        legend_title_text="Ward groups",
        margin=dict(l=10, r=10, t=25, b=10)
    )

    st.plotly_chart(
        group_chart,
        use_container_width=True
    )

    st.subheader("Compare ward groups")

    summary_display = group_summary.reset_index().copy()

    summary_display["Ward group"] = (
        summary_display["kmeans_cluster"].map(group_names)
    )

    summary_display = summary_display.rename(
        columns={
            "number_of_wards": "Number of wards",
            "average_waste": "Average monthly waste (kg)",
            "average_trend": "Average monthly change (kg)"
        }
    )

    st.dataframe(
        summary_display[
            [
                "Ward group",
                "Number of wards",
                "Average monthly waste (kg)",
                "Average monthly change (kg)"
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Ward details")

    ward_display = clustered[
        [
            "ward",
            "ward_group",
            "avg_monthly_waste",
            "trend_slope",
            "growth_pct"
        ]
    ].copy()

    ward_display = ward_display.rename(
        columns={
            "ward": "Ward",
            "ward_group": "Ward group",
            "avg_monthly_waste": "Average monthly waste (kg)",
            "trend_slope": "Monthly change (kg)",
            "growth_pct": "Change from first to last observation (%)"
        }
    )

    st.dataframe(
        ward_display.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download ward analysis",
        data=ward_display.to_csv(index=False).encode("utf-8"),
        file_name="ward_analysis.csv",
        mime="text/csv"
    )

    with st.expander("Technical details for project evaluation"):

        st.write(
            "The system uses K-Means to group wards based on "
            "standardized numerical features."
        )

        st.write(
            "DBSCAN is also applied to identify density-based "
            "groups and observations classified as noise."
        )

        if silhouette is not None:
            st.write(
                f"Group separation score (silhouette): "
                f"{silhouette:.3f}"
            )

        st.write(
            "DBSCAN noise points are not automatically problem wards. "
            "They require separate investigation."
        )

        st.write(
            f"DBSCAN noise observations: "
            f"{int((clustered['dbscan_cluster'] == -1).sum())}"
        )


# ==========================================================
# 10. WASTE FORECAST
# ==========================================================

with forecast_tab:

    st.subheader("Estimate future waste quantities")

    st.write(
        "The forecasting model learns from previous monthly records "
        "and estimates waste for later months."
    )

    try:

        model, metrics, predictions = train_and_evaluate(monthly)

        a, b, c = st.columns(3)

        a.metric(
            "Average prediction error",
            f"{metrics['MAE']:,.0f} kg"
        )

        b.metric(
            "Large-error-sensitive measure",
            f"{metrics['RMSE']:,.0f} kg"
        )

        c.metric(
            "Records evaluated",
            f"{metrics['testing_records']:,}"
        )

        st.caption(
            "Lower prediction errors are generally better. "
            "These results must be compared with a simple baseline "
            "before judging forecast quality."
        )

        with st.expander("What do these accuracy measures mean?"):

            st.write(
                "**Average prediction error:** the average absolute "
                "difference between predicted and actual waste."
            )

            st.write(
                "**Large-error-sensitive measure:** a metric that "
                "penalizes larger mistakes more strongly."
            )

            st.write(
                "These measurements describe performance on held-out "
                "records. They do not guarantee future accuracy."
            )

        forecast_ward = st.selectbox(
            "Choose a ward to inspect",
            sorted(predictions["ward"].unique()),
            key="forecast_ward"
        )

        selected_predictions = predictions[
            predictions["ward"] == forecast_ward
        ].sort_values("month")

        if not selected_predictions.empty:

            forecast_long = selected_predictions.melt(
                id_vars=["month"],
                value_vars=[
                    "waste_kg",
                    "predicted_waste_kg"
                ],
                var_name="series",
                value_name="quantity"
            )

            forecast_long["series"] = (
                forecast_long["series"].map({
                    "waste_kg": "Actual recorded waste",
                    "predicted_waste_kg": "Estimated waste"
                })
            )

            fig_forecast = px.line(
                forecast_long,
                x="month",
                y="quantity",
                color="series",
                markers=True,
                labels={
                    "month": "Month",
                    "quantity": "Waste (kg)",
                    "series": "Measurement"
                }
            )

            st.plotly_chart(
                fig_forecast,
                use_container_width=True
            )

        st.download_button(
            "Download forecast report",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="forecast_results.csv",
            mime="text/csv"
        )

    except Exception as error:

        st.error(
            f"The forecast could not be generated: {error}"
        )

        st.caption(
            "Check the available historical records and "
            "the forecasting module."
        )


# ==========================================================
# 11. ATTENTION REQUIRED
# ==========================================================

with attention_tab:

    st.subheader("Records that may need checking")

    st.write(
        "These records differ from common patterns in the dataset. "
        "Review them before deciding whether corrective action is needed."
    )

    st.metric(
        "Records flagged for review",
        f"{len(flagged):,}"
    )

    if not flagged.empty:

        fig_anomalies = px.scatter(
            flagged,
            x="date",
            y="waste_kg",
            color="ward",
            hover_data=["anomaly_score"],
            labels={
                "date": "Date",
                "waste_kg": "Waste recorded (kg)",
                "ward": "Ward"
            }
        )

        st.plotly_chart(
            fig_anomalies,
            use_container_width=True
        )

        anomaly_display = flagged.copy()

        anomaly_display = anomaly_display.rename(
            columns={
                "date": "Date",
                "ward": "Ward",
                "waste_kg": "Recorded waste (kg)",
                "anomaly_score": "Unusualness score"
            }
        )

        st.dataframe(
            anomaly_display,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download records for review",
            data=flagged.to_csv(index=False).encode("utf-8"),
            file_name="records_for_review.csv",
            mime="text/csv"
        )

    else:

        st.success(
            "No records were flagged with the current settings."
        )

    st.caption(
        "A flagged record is not proof of a data error, "
        "missed collection, or sanitation issue."
    )


# ==========================================================
# 12. FOOTER
# ==========================================================

st.divider()

st.caption(
    "Smart Municipal Waste Management | "
    "Ward analysis, forecasting and record review"
)

st.caption(
    "For academic demonstration. Verify the data and "
    "validate the models before operational use."
)
