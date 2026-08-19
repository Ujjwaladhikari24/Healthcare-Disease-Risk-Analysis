import os
import glob
import json
import warnings

import joblib
import numpy as np
import pandas as pd

import matplotlib

# Prevent Tkinter / GUI errors on Windows
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "Data")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
STATIC_DIR = os.path.join(BASE_DIR, "Static")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_json(filename, data):

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


def calculate_metrics(y_test, predictions):

    return {
        "accuracy": round(
            accuracy_score(
                y_test,
                predictions
            ) * 100,
            2
        ),

        "precision": round(
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ) * 100,
            2
        ),

        "recall": round(
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ) * 100,
            2
        ),

        "f1_score": round(
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ) * 100,
            2
        )
    }


def save_confusion_matrix(
    y_test,
    predictions,
    title,
    filename
):

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(
        figsize=(7, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            STATIC_DIR,
            filename
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close("all")


def train_numeric_model(
    X,
    y,
    model_name,
    model_filename,
    info_filename,
    confusion_filename
):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    preprocessing = Pipeline([
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
    ])

    model = Pipeline([
        (
            "preprocessing",
            preprocessing
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    save_confusion_matrix(
        y_test,
        predictions,
        model_name + " - Confusion Matrix",
        confusion_filename
    )

    joblib.dump(
        model,
        os.path.join(
            MODEL_DIR,
            model_filename
        )
    )

    info = {
        "name": model_name,

        "features": list(
            X.columns
        ),

        "metrics": metrics,

        "classes": [
            str(x)
            for x in model.classes_
        ]
    }

    save_json(
        info_filename,
        info
    )

    return model, info


def train_mixed_model(
    X,
    y,
    model_name,
    model_filename,
    info_filename,
    confusion_filename
):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    numeric_columns = list(
        X.select_dtypes(
            include=[
                "int64",
                "int32",
                "float64",
                "float32"
            ]
        ).columns
    )

    categorical_columns = list(
        X.select_dtypes(
            include=[
                "object",
                "category"
            ]
        ).columns
    )

    transformers = []

    if numeric_columns:

        numeric_pipeline = Pipeline([
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
        ])

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            )
        )

    if categorical_columns:

        categorical_pipeline = Pipeline([
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
        ])

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    preprocessing = ColumnTransformer(
        transformers=transformers
    )

    model = Pipeline([
        (
            "preprocessing",
            preprocessing
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    metrics = calculate_metrics(
        y_test,
        predictions
    )

    save_confusion_matrix(
        y_test,
        predictions,
        model_name + " - Confusion Matrix",
        confusion_filename
    )

    joblib.dump(
        model,
        os.path.join(
            MODEL_DIR,
            model_filename
        )
    )

    info = {
        "name": model_name,

        "features": list(
            X.columns
        ),

        "metrics": metrics,

        "classes": [
            str(x)
            for x in model.classes_
        ]
    }

    save_json(
        info_filename,
        info
    )

    return model, info


# ============================================================
# FIND ALL DATA FILES
# ============================================================

all_files = glob.glob(
    os.path.join(
        DATA_DIR,
        "*"
    )
)

data_files = [
    file
    for file in all_files
    if os.path.isfile(file)
]


if not data_files:

    raise FileNotFoundError(
        "No datasets were found inside the Data folder."
    )


print()
print("=" * 70)
print("HEALTHCARE DISEASE RISK ANALYSIS")
print("=" * 70)

print()
print("Datasets found:")

for file in data_files:

    print(
        " -",
        os.path.basename(file)
    )


# ============================================================
# 1. DIABETES MODEL
# ============================================================

print()
print("=" * 70)
print("1/3  TRAINING DIABETES MODEL")
print("=" * 70)


diabetes_path = None


for file in data_files:

    filename = os.path.basename(
        file
    ).lower()

    if (
        "diabetes" in filename
        and file.lower().endswith(".csv")
    ):

        diabetes_path = file
        break


if diabetes_path is None:

    raise FileNotFoundError(
        "Diabetes CSV file was not found."
    )


print(
    "Dataset:",
    os.path.basename(
        diabetes_path
    )
)


diabetes = pd.read_csv(
    diabetes_path
)


DIABETES_TARGET = "Diabetes_012"


if DIABETES_TARGET not in diabetes.columns:

    raise ValueError(
        "The diabetes dataset does not contain "
        "'Diabetes_012'."
    )


diabetes = diabetes.dropna(
    subset=[
        DIABETES_TARGET
    ]
)


X_diabetes = diabetes.drop(
    columns=[
        DIABETES_TARGET
    ]
)

y_diabetes = diabetes[
    DIABETES_TARGET
].astype(int)


diabetes_model, diabetes_info = train_numeric_model(
    X_diabetes,
    y_diabetes,
    "Diabetes Risk Model",
    "diabetes_model.pkl",
    "diabetes_info.json",
    "diabetes_confusion_matrix.png"
)


print(
    "Diabetes Accuracy:",
    diabetes_info["metrics"]["accuracy"],
    "%"
)


# ============================================================
# DIABETES DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(7, 5)
)

diabetes[
    DIABETES_TARGET
].value_counts().sort_index().plot(
    kind="bar"
)

plt.title(
    "Diabetes Class Distribution"
)

plt.xlabel(
    "Diabetes Category"
)

plt.ylabel(
    "Number of Patients"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        STATIC_DIR,
        "diabetes_distribution.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close("all")


# ============================================================
# DIABETES FEATURE IMPORTANCE
# ============================================================

classifier = (
    diabetes_model
    .named_steps[
        "classifier"
    ]
)

importance = (
    classifier
    .feature_importances_
)


importance_df = pd.DataFrame({

    "feature":
        X_diabetes.columns,

    "importance":
        importance

})


importance_df = (
    importance_df
    .sort_values(
        "importance",
        ascending=False
    )
    .head(15)
)


plt.figure(
    figsize=(9, 6)
)

plt.barh(
    importance_df["feature"][::-1],
    importance_df["importance"][::-1]
)

plt.title(
    "Top Diabetes Risk Features"
)

plt.xlabel(
    "Feature Importance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        STATIC_DIR,
        "feature_importance.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close("all")


# ============================================================
# 2. HEART DISEASE MODEL
# ============================================================

print()
print("=" * 70)
print("2/3  TRAINING HEART DISEASE MODEL")
print("=" * 70)


heart_path = None


for file in data_files:

    filename = os.path.basename(
        file
    ).lower()

    if (
        "cleveland" in filename
        or "heart" in filename
    ):

        heart_path = file
        break


if heart_path is None:

    raise FileNotFoundError(
        "Heart disease dataset was not found."
    )


print(
    "Dataset:",
    os.path.basename(
        heart_path
    )
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
    header=None,
    names=heart_columns
)


heart = heart.replace(
    "?",
    np.nan
)


for column in heart.columns:

    heart[column] = pd.to_numeric(
        heart[column],
        errors="coerce"
    )


heart = heart.dropna(
    subset=[
        "target"
    ]
)


heart["target"] = (
    heart["target"] > 0
).astype(int)


X_heart = heart.drop(
    columns=[
        "target"
    ]
)

y_heart = heart[
    "target"
]


heart_model, heart_info = train_numeric_model(
    X_heart,
    y_heart,
    "Heart Disease Risk Model",
    "heart_disease_model.pkl",
    "heart_disease_info.json",
    "heart_disease_confusion_matrix.png"
)


print(
    "Heart Disease Accuracy:",
    heart_info["metrics"]["accuracy"],
    "%"
)


# ============================================================
# 3. STROKE MODEL
# ============================================================

print()
print("=" * 70)
print("3/3  TRAINING STROKE MODEL")
print("=" * 70)


stroke_path = None


for file in data_files:

    filename = os.path.basename(
        file
    ).lower()

    if (
        "stroke" in filename
        and file.lower().endswith(".csv")
    ):

        stroke_path = file
        break


if stroke_path is None:

    raise FileNotFoundError(
        "Stroke CSV file was not found."
    )


print(
    "Dataset:",
    os.path.basename(
        stroke_path
    )
)


stroke = pd.read_csv(
    stroke_path
)


if "stroke" not in stroke.columns:

    raise ValueError(
        "The stroke dataset does not contain "
        "'stroke'."
    )


stroke = stroke.drop(
    columns=[
        "id"
    ],
    errors="ignore"
)


stroke = stroke.dropna(
    subset=[
        "stroke"
    ]
)


X_stroke = stroke.drop(
    columns=[
        "stroke"
    ]
)

y_stroke = stroke[
    "stroke"
].astype(int)


stroke_model, stroke_info = train_mixed_model(
    X_stroke,
    y_stroke,
    "Stroke Risk Model",
    "stroke_model.pkl",
    "stroke_info.json",
    "stroke_confusion_matrix.png"
)


print(
    "Stroke Accuracy:",
    stroke_info["metrics"]["accuracy"],
    "%"
)


# ============================================================
# SAVE COMPLETE PROJECT INFORMATION
# ============================================================

summary = {

    "project":
        "Healthcare Disease Risk Analysis",

    "description":
        "Multi-disease healthcare risk analysis using machine learning.",

    "models": {

        "diabetes":
            diabetes_info,

        "heart_disease":
            heart_info,

        "stroke":
            stroke_info

    }

}


save_json(
    "model_summary.json",
    summary
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("ALL THREE MODELS TRAINED SUCCESSFULLY")
print("=" * 70)

print()
print("Generated model files:")

print(
    "Models/diabetes_model.pkl"
)

print(
    "Models/heart_disease_model.pkl"
)

print(
    "Models/stroke_model.pkl"
)

print()
print("Generated information files:")

print(
    "Models/diabetes_info.json"
)

print(
    "Models/heart_disease_info.json"
)

print(
    "Models/stroke_info.json"
)

print(
    "Models/model_summary.json"
)

print()
print("Generated visualizations:")

print(
    "Static/diabetes_distribution.png"
)

print(
    "Static/feature_importance.png"
)

print(
    "Static/diabetes_confusion_matrix.png"
)

print(
    "Static/heart_disease_confusion_matrix.png"
)

print(
    "Static/stroke_confusion_matrix.png"
)

print()
print("=" * 70)
print("STEP 2 COMPLETE")
print("=" * 70)