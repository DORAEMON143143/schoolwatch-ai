"""Synthetic training data and model helpers for the SchoolWatch AI demo."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS: Final[list[str]] = [
    "temperature_c",
    "smoke_level",
    "co_ppm",
    "gas_level",
    "sound_db",
    "crowd_count",
    "motion_index",
    "flame_detected",
    "event_duration_min",
]

CLASS_NAMES: Final[dict[int, str]] = {
    0: "Safe",
    1: "Normal Alert",
    2: "High-Priority Emergency",
}

CLASS_COLORS: Final[dict[str, str]] = {
    "Safe": "#168568",
    "Normal Alert": "#d68a24",
    "High-Priority Emergency": "#c54d4d",
}


def generate_training_data(rows_per_class: int = 900, seed: int = 42) -> pd.DataFrame:
    """Create reproducible, intentionally overlapping synthetic sensor readings."""
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int | str]] = []

    profiles = {
        0: {
            "temperature_c": (27, 4.5),
            "smoke_level": (6, 5),
            "co_ppm": (5, 4),
            "gas_level": (18, 12),
            "sound_db": (61, 13),
            "crowd_count": (28, 17),
            "motion_index": (35, 18),
            "event_duration_min": (1.5, 1),
        },
        1: {
            "temperature_c": (37, 7),
            "smoke_level": (34, 18),
            "co_ppm": (22, 14),
            "gas_level": (55, 24),
            "sound_db": (83, 18),
            "crowd_count": (49, 21),
            "motion_index": (63, 21),
            "event_duration_min": (5, 3),
        },
        2: {
            "temperature_c": (53, 10),
            "smoke_level": (73, 20),
            "co_ppm": (43, 19),
            "gas_level": (91, 24),
            "sound_db": (103, 17),
            "crowd_count": (67, 19),
            "motion_index": (80, 16),
            "event_duration_min": (9, 4),
        },
    }

    for severity, profile in profiles.items():
        for _ in range(rows_per_class):
            record: dict[str, float | int | str] = {
                name: round(float(np.clip(rng.normal(mean, std), low, high)), 2)
                for name, mean, std, low, high in (
                    ("temperature_c", *profile["temperature_c"], 15, 85),
                    ("smoke_level", *profile["smoke_level"], 0, 100),
                    ("co_ppm", *profile["co_ppm"], 0, 100),
                    ("gas_level", *profile["gas_level"], 0, 120),
                    ("sound_db", *profile["sound_db"], 25, 130),
                    ("crowd_count", *profile["crowd_count"], 0, 120),
                    ("motion_index", *profile["motion_index"], 0, 100),
                    ("event_duration_min", *profile["event_duration_min"], 0.1, 30),
                )
            }
            record["crowd_count"] = int(round(float(record["crowd_count"])))
            record["flame_detected"] = int(
                rng.random() < ({0: 0.002, 1: 0.04, 2: 0.35}[severity])
            )
            record["severity"] = severity
            record["severity_label"] = CLASS_NAMES[severity]
            records.append(record)

    return pd.DataFrame.from_records(records)


def build_model(seed: int = 42) -> tuple[Pipeline, dict[str, object], pd.DataFrame]:
    """Train and evaluate the Logistic Regression model on held-out synthetic data."""
    data = generate_training_data(seed=seed)
    features = data[FEATURE_COLUMNS]
    target = data["severity"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=seed,
        stratify=target,
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "report": classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=[CLASS_NAMES[i] for i in range(3)],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1, 2]),
        "test_size": len(y_test),
    }
    return model, metrics, data


def classify_reading(model: Pipeline, reading: dict[str, float | int]) -> dict[str, object]:
    """Return a model prediction and per-class scores for one sensor reading."""
    frame = pd.DataFrame([{name: reading[name] for name in FEATURE_COLUMNS}])
    prediction = int(model.predict(frame)[0])
    probabilities = model.predict_proba(frame)[0]
    scores = {CLASS_NAMES[int(label)]: float(score) for label, score in zip(model.classes_, probabilities)}
    return {
        "severity": prediction,
        "status": CLASS_NAMES[prediction],
        "scores": scores,
    }


def generate_reading(scenario: str, seed: int | None = None) -> dict[str, float | int | str]:
    """Generate one plausible sensor snapshot for a named demo scenario."""
    rng = np.random.default_rng(seed)
    scenarios = {
        "Routine activity": (0, "Safe / Normal Activity"),
        "Minor smoke": (1, "Minor Smoke"),
        "Crowd disturbance": (1, "Crowd Disturbance"),
        "Possible gas issue": (1, "Possible Gas Issue"),
        "Fire emergency": (2, "Fire Emergency"),
        "Severe heat event": (2, "Severe Heat Emergency"),
    }
    severity, incident_type = scenarios[scenario]
    data = generate_training_data(rows_per_class=1, seed=int(rng.integers(0, 2**31)))
    row = data[data["severity"] == severity].iloc[0]
    reading: dict[str, float | int | str] = {
        name: int(row[name]) if name == "crowd_count" else float(row[name])
        for name in FEATURE_COLUMNS
        if name != "flame_detected"
    }
    reading["flame_detected"] = int(row["flame_detected"])
    if scenario == "Fire emergency":
        reading.update({"temperature_c": 71.0, "smoke_level": 92.0, "flame_detected": 1})
    elif scenario == "Routine activity":
        reading.update({"temperature_c": 25.0, "smoke_level": 3.0, "co_ppm": 4.0, "gas_level": 16.0})
    reading["incident_type"] = incident_type
    return reading
