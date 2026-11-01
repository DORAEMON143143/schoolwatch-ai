# SchoolWatch AI

An internship-ready Streamlit prototype demonstrating a simulated school-safety monitoring workflow:

**sensor readings → multiclass Logistic Regression → alert action → automatic clear → history**

## Features

- Overview dashboard with current status and session activity
- Manual sensor assessment with nine model input features
- Simulated routine, smoke, crowd, gas, fire, and heat scenarios
- Safe, Normal Alert, and High-Priority Emergency predictions
- Automatic `CLEAR` action when a safe reading follows an active alert
- Held-out model metrics and confusion matrix
- Downloadable session history as CSV
- Synthetic, reproducible data generation; no external dataset or sensor hardware required

## Run locally

Requires Python 3.10 or later.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Project structure

```text
app.py             Streamlit dashboard and interaction flows
safety_model.py    Synthetic data, Logistic Regression, and prediction helpers
requirements.txt   Python dependencies
tests/             Model and alert workflow checks
```

## Model and data notes

The app trains a scaled multiclass Logistic Regression model from generated sensor profiles when it starts. Inputs include temperature, smoke, CO, gas, sound, crowd count, motion, flame detection, and event duration. Data is intentionally synthetic and labels are generated from scenario profiles; displayed accuracy is a teaching/demo metric and must not be interpreted as evidence of real-world performance.

This is an academic prototype, not a life-safety system. It does not connect to real sensors, contact emergency services, or replace trained human judgment. Real deployments require validated data and sensors, safety engineering, privacy review, fail-safe design, human oversight, and extensive testing.

## Suggested internship report sections

1. Introduction and problem statement
2. Objectives, scope, and limitations
3. Synthetic dataset and sensor features
4. Preprocessing and Logistic Regression
5. Train/test split and evaluation
6. Prediction and alert logic
7. Dashboard, simulation, and history
8. Results, ethical considerations, and future work

## Future extensions

Real validated datasets, room-wise monitoring, authenticated staff acknowledgements, secure database persistence, and IoT integrations can be explored as separate, safety-reviewed research work.
