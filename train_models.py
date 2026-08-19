import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# FOLDERS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "Data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "Models"
)

STATIC_DIR = os.path.join(
    BASE_DIR,
    "Static"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_metrics(
    name,
    model,
    X_test,
    y_test,
    predictions
):

    metrics = {}

    metrics["accuracy"] = float(
        accuracy_score(
            y_test,
            predictions
        )
    )

    metrics["balanced_accuracy"] = float(
        balanced_accuracy_score(
            y_test,
            predictions
        )
    )

    metrics["precision"] = float(
        precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    metrics["recall"] = float(
        recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    metrics["f1_score"] = float(
        f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    try:

        probabilities = model.predict_proba(
            X_test
        )

        if len(
            np.unique(y_test)
        ) == 2:

            metrics["roc_auc"] = float(
                roc_auc_score(
                    y_test,
                    probabilities[:, 1]
                )
            )

        else:

            metrics["roc_auc"] = float(
                roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr"
                )
            )

    except Exception:

        metrics["roc_auc"] = None


    print()
    print("=" * 60)
    print(name.upper())
    print("=" * 60)

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print(
        "Accuracy:",
        round(metrics["accuracy"] * 100, 2),
        "%"
    )

    print(
        "Precision:",
        round(metrics["precision"] * 100, 2),
        "%"
    )

    print(
        "Recall:",
        round(metrics["recall"] * 100, 2),
        "%"
    )

    print(
        "F1 Score:",
        round(metrics["f1_score"] * 100, 2),
        "%"
    )

    if metrics["roc_auc"] is not None:

        print(
            "ROC-AUC:",
            round(metrics["roc_auc"], 4)
        )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(
        f"{name} - Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            STATIC_DIR,
            f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
        )
    )

    plt.close()


    return metrics


# ============================================================
# HEART DISEASE MODEL
# ============================================================

print()
print("=" * 60)
print("TRAINING HEART DISEASE MODEL")
print("=" * 60)


heart_path = os.path.join(
    DATA_DIR,
    "processed.cleveland.data"
)


heart_columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]


heart = pd.read_csv(
    heart_path,
    names=heart_columns,
    na_values="?"
)


# Convert everything to numeric

for column in heart.columns:

    heart[column] = pd.to_numeric(
        heart[column],
        errors="coerce"
    )


# Original UCI target:
# 0 = no disease
# 1-4 = disease
heart["target"] = (
    heart["target"] > 0
).astype(int)


X_heart = heart.drop(
    columns=["target"]
)

y_heart = heart["target"]


heart_numeric = list(
    X_heart.columns
)


heart_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            ),
            heart_numeric
        )
    ]
)


heart_model = Pipeline(
    steps=[
        (
            "preprocessor",
            heart_preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=250,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X_heart,
    y_heart,
    test_size=0.20,
    random_state=42,
    stratify=y_heart
)


heart_model.fit(
    X_train,
    y_train
)


heart_predictions = heart_model.predict(
    X_test
)


heart_metrics = save_metrics(
    "Heart Disease",
    heart_model,
    X_test,
    y_test,
    heart_predictions
)


joblib.dump(
    heart_model,
    os.path.join(
        MODEL_DIR,
        "heart_disease_model.pkl"
    )
)


with open(
    os.path.join(
        MODEL_DIR,
        "heart_disease_info.json"
    ),
    "w"
) as file:

    json.dump(
        {
            "model": "Random Forest",
            "disease": "Heart Disease",
            "features": heart_columns[:-1],
            "metrics": heart_metrics
        },
        file,
        indent=4
    )


# ============================================================
# STROKE MODEL
# ============================================================

print()
print("=" * 60)
print("TRAINING STROKE MODEL")
print("=" * 60)


stroke_path = os.path.join(
    DATA_DIR,
    "healthcare-dataset-stroke-data.csv"
)


stroke = pd.read_csv(
    stroke_path
)


# Remove ID because it has no predictive meaning

if "id" in stroke.columns:

    stroke = stroke.drop(
        columns=["id"]
    )


# ------------------------------------------------------------
# TARGET
# ------------------------------------------------------------

stroke["stroke"] = pd.to_numeric(
    stroke["stroke"],
    errors="coerce"
)


stroke = stroke.dropna(
    subset=["stroke"]
)


X_stroke = stroke.drop(
    columns=["stroke"]
)

y_stroke = stroke["stroke"].astype(int)


# ------------------------------------------------------------
# FEATURE TYPES
# ------------------------------------------------------------

categorical_features = X_stroke.select_dtypes(
    include=["object"]
).columns.tolist()


numeric_features = X_stroke.select_dtypes(
    exclude=["object"]
).columns.tolist()


# ------------------------------------------------------------
# PREPROCESSOR
# ------------------------------------------------------------

stroke_numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


stroke_categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


stroke_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            stroke_numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            stroke_categorical_pipeline,
            categorical_features
        )
    ]
)


stroke_model = Pipeline(
    steps=[
        (
            "preprocessor",
            stroke_preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=250,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X_stroke,
    y_stroke,
    test_size=0.20,
    random_state=42,
    stratify=y_stroke
)


stroke_model.fit(
    X_train,
    y_train
)


stroke_predictions = stroke_model.predict(
    X_test
)


stroke_metrics = save_metrics(
    "Stroke",
    stroke_model,
    X_test,
    y_test,
    stroke_predictions
)


joblib.dump(
    stroke_model,
    os.path.join(
        MODEL_DIR,
        "stroke_model.pkl"
    )
)


with open(
    os.path.join(
        MODEL_DIR,
        "stroke_info.json"
    ),
    "w"
) as file:

    json.dump(
        {
            "model": "Random Forest",
            "disease": "Stroke",
            "features": X_stroke.columns.tolist(),
            "metrics": stroke_metrics
        },
        file,
        indent=4
    )


# ============================================================
# DATASET STATISTICS
# ============================================================

dataset_statistics = {

    "heart_disease": {
        "records": int(len(heart)),
        "features": int(
            len(X_heart.columns)
        ),
        "disease_cases": int(
            y_heart.sum()
        ),
        "no_disease_cases": int(
            (y_heart == 0).sum()
        )
    },

    "stroke": {
        "records": int(len(stroke)),
        "features": int(
            len(X_stroke.columns)
        ),
        "stroke_cases": int(
            y_stroke.sum()
        ),
        "no_stroke_cases": int(
            (y_stroke == 0).sum()
        )
    }

}


with open(
    os.path.join(
        MODEL_DIR,
        "dataset_statistics.json"
    ),
    "w"
) as file:

    json.dump(
        dataset_statistics,
        file,
        indent=4
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print()
print("Created files:")

print(
    "Models/heart_disease_model.pkl"
)

print(
    "Models/heart_disease_info.json"
)

print(
    "Models/stroke_model.pkl"
)

print(
    "Models/stroke_info.json"
)

print(
    "Models/dataset_statistics.json"
)

print()
print("Your Heart Disease and Stroke models are ready.")