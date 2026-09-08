# AI Salary Hike Predictor

A small Streamlit application for HR teams. It reads employee data from CSV files, trains an AI regression model on historical performance and hike data, and predicts salary hike percentages for employees.

## Features

- Upload historical employee CSV data.
- Select rating, salary, tenure, department, and other feature columns.
- Train a Random Forest regression model.
- Predict salary hike percentage, hike amount, and new salary.
- Download the final predictions as CSV.
- Includes `sample_employee_data.csv` for quick testing.

## CSV Columns

Recommended columns:

- `employee_id`
- `name`
- `department`
- `current_salary`
- `performance_rating`
- `years_at_company`
- `last_hike_percent`
- `training_score`
- `actual_hike_percent`

The `actual_hike_percent` column is the historical target value used to train the model.

## Run The App

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Streamlit:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Notes For HR Use

This app should support review, not replace HR judgment. Salary decisions should also consider budget, role benchmarks, internal equity, promotion status, and company policy.
