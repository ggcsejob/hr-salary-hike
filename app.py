import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(page_title="AI Salary Hike Predictor", page_icon=":chart_with_upwards_trend:", layout="wide")

APP_DIR = Path(__file__).parent
SAMPLE_FILE = APP_DIR / "sample_employee_data.csv"
DEFAULT_FEATURES = [
    "current_salary",
    "performance_rating",
    "years_at_company",
    "last_hike_percent",
    "training_score",
    "department",
]
DEFAULT_TARGET = "actual_hike_percent"


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_FILE)


def read_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def build_model(df: pd.DataFrame, feature_columns: list[str], target_column: str):
    model_df = df[feature_columns + [target_column]].dropna()
    if len(model_df) < 8:
        raise ValueError("At least 8 complete rows are needed to train the model.")

    x = model_df[feature_columns]
    y = model_df[target_column]

    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in feature_columns if column not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=250,
                    min_samples_leaf=2,
                    random_state=42,
                ),
            ),
        ]
    )

    if len(model_df) >= 15:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=42
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics = {
            "MAE": float(mean_absolute_error(y_test, predictions)),
            "R2": float(r2_score(y_test, predictions)),
            "test_rows": len(y_test),
        }
        model.fit(x, y)
    else:
        model.fit(x, y)
        metrics = {"MAE": None, "R2": None, "test_rows": 0}

    return model, metrics, model_df


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def add_predictions(df: pd.DataFrame, model, feature_columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    prediction_input = result[feature_columns]
    result["predicted_hike_percent"] = np.clip(model.predict(prediction_input), 0, 35).round(2)

    if "current_salary" in result.columns:
        result["predicted_hike_amount"] = (
            result["current_salary"] * result["predicted_hike_percent"] / 100
        ).round(2)
        result["predicted_new_salary"] = (
            result["current_salary"] + result["predicted_hike_amount"]
        ).round(2)

    return result


st.title("AI Salary Hike Predictor")
st.caption("Upload employee CSV data, train on historical hike decisions, and estimate salary hikes for HR review.")

sample_data = load_sample_data()

with st.sidebar:
    st.header("Data")
    mode = st.radio(
        "Training data",
        ["Use sample historical data", "Upload historical CSV"],
        label_visibility="collapsed",
    )

    historical_file = None
    if mode == "Upload historical CSV":
        historical_file = st.file_uploader("Historical employee CSV", type=["csv"])

    prediction_file = st.file_uploader("Employees to predict CSV", type=["csv"])

    st.download_button(
        "Download sample CSV",
        data=sample_data.to_csv(index=False),
        file_name="sample_employee_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

try:
    historical_data = read_csv(historical_file) if historical_file else sample_data
except Exception as exc:
    st.error(f"Could not read historical CSV: {exc}")
    st.stop()

try:
    prediction_data = read_csv(prediction_file) if prediction_file else historical_data.drop(columns=[DEFAULT_TARGET], errors="ignore")
except Exception as exc:
    st.error(f"Could not read prediction CSV: {exc}")
    st.stop()

st.subheader("Configure Model")
left, right = st.columns([2, 1])

with left:
    available_columns = historical_data.columns.tolist()
    default_features = [column for column in DEFAULT_FEATURES if column in available_columns]
    feature_columns = st.multiselect(
        "Feature columns used by the AI model",
        available_columns,
        default=default_features or available_columns[: min(5, len(available_columns))],
    )

with right:
    target_candidates = [
        column
        for column in available_columns
        if column not in feature_columns and pd.api.types.is_numeric_dtype(historical_data[column])
    ]
    default_target_index = target_candidates.index(DEFAULT_TARGET) if DEFAULT_TARGET in target_candidates else 0
    target_column = st.selectbox(
        "Known hike column",
        target_candidates,
        index=default_target_index if target_candidates else None,
        help="This should contain historical approved hike percentage values.",
    )

missing_prediction_columns = [column for column in feature_columns if column not in prediction_data.columns]
if not feature_columns:
    st.warning("Choose at least one feature column.")
    st.stop()

if not target_column:
    st.warning("Choose a numeric target column such as actual_hike_percent.")
    st.stop()

if missing_prediction_columns:
    st.error("Prediction CSV is missing these feature columns: " + ", ".join(missing_prediction_columns))
    st.stop()

try:
    model, metrics, training_rows = build_model(historical_data, feature_columns, target_column)
    predictions = add_predictions(prediction_data, model, feature_columns)
except Exception as exc:
    st.error(str(exc))
    st.stop()

metric_cols = st.columns(4)
metric_cols[0].metric("Training Rows", f"{len(training_rows):,}")
metric_cols[1].metric("Employees Predicted", f"{len(predictions):,}")
metric_cols[2].metric("Avg Hike", f"{predictions['predicted_hike_percent'].mean():.2f}%")
if metrics["MAE"] is not None:
    metric_cols[3].metric("Model MAE", f"{metrics['MAE']:.2f}%")
else:
    metric_cols[3].metric("Model MAE", "Need 15+ rows")

st.subheader("Predicted Salary Hikes")

display_columns = [
    column
    for column in [
        "employee_id",
        "name",
        "department",
        "current_salary",
        "performance_rating",
        "predicted_hike_percent",
        "predicted_hike_amount",
        "predicted_new_salary",
    ]
    if column in predictions.columns
]

st.dataframe(
    predictions[display_columns] if display_columns else predictions,
    use_container_width=True,
    hide_index=True,
)

if "predicted_hike_amount" in predictions.columns:
    total_budget = predictions["predicted_hike_amount"].sum()
    average_new_salary = predictions["predicted_new_salary"].mean()
    budget_cols = st.columns(2)
    budget_cols[0].metric("Estimated Hike Budget", format_currency(total_budget))
    budget_cols[1].metric("Average New Salary", format_currency(average_new_salary))

csv_buffer = io.StringIO()
predictions.to_csv(csv_buffer, index=False)
st.download_button(
    "Download predictions",
    data=csv_buffer.getvalue(),
    file_name="salary_hike_predictions.csv",
    mime="text/csv",
    type="primary",
)

with st.expander("CSV format guide"):
    st.write(
        "For best results, include employee identifier columns plus numeric fields such as "
        "current_salary, performance_rating, years_at_company, last_hike_percent, "
        "training_score, and a historical target column such as actual_hike_percent."
    )
