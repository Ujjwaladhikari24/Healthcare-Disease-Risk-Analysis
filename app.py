import os
import traceback

import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request


# ============================================================
# APPLICATION SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "Models"
)

app = Flask(
    __name__,
    template_folder="Templates",
    static_folder="Static"
)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(names):

    for name in names:

        path = os.path.join(
            MODEL_DIR,
            name
        )

        if os.path.exists(path):

            try:

                model = joblib.load(path)

                print(
                    f"[OK] Loaded model: {name}"
                )

                return model

            except Exception as e:

                print(
                    f"[ERROR] Could not load {name}: {e}"
                )

    return None


diabetes_model = load_model([
    "diabetes_model.pkl",
    "diabetes.pkl",
    "diabetes_model.joblib"
])


heart_model = load_model([
    "heart_disease_model.pkl",
    "heart_model.pkl",
    "heart_disease.pkl",
    "heart_model.joblib"
])


stroke_model = load_model([
    "stroke_model.pkl",
    "stroke.pkl",
    "stroke_model.joblib"
])


# ============================================================
# OPTIONAL SCALER
# ============================================================

scaler = None

scaler_path = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

if os.path.exists(scaler_path):

    try:

        scaler = joblib.load(
            scaler_path
        )

        print("[OK] Scaler loaded.")

    except Exception as e:

        print(
            "[WARNING] Scaler could not be loaded:",
            e
        )


# ============================================================
# PRINT MODEL INFORMATION
# ============================================================

print()
print("=" * 65)
print("HEALTHCARE DISEASE RISK ANALYSIS")
print("=" * 65)

print(
    "Diabetes model:",
    "AVAILABLE" if diabetes_model is not None else "MISSING"
)

print(
    "Heart disease model:",
    "AVAILABLE" if heart_model is not None else "MISSING"
)

print(
    "Stroke model:",
    "AVAILABLE" if stroke_model is not None else "MISSING"
)

print("=" * 65)
print()


# ============================================================
# GENERAL HELPERS
# ============================================================

def get_value(*names, default=None):

    for name in names:

        value = request.form.get(name)

        if value is not None:

            value = str(value).strip()

            if value != "":

                return value

    return default


def to_float(value, default=0.0):

    try:

        if value is None:
            return default

        value = str(value).strip()

        if value == "":
            return default

        return float(value)

    except Exception:

        return default


def to_int(value, default=0):

    try:

        if value is None:
            return default

        value = str(value).strip()

        if value == "":
            return default

        return int(float(value))

    except Exception:

        return default


def clean_number(value, default=0):

    value = to_float(
        value,
        default
    )

    if not np.isfinite(value):

        return default

    return value


def clean_text(value, default="Unknown"):

    if value is None:

        return default

    value = str(value).strip()

    if value == "":

        return default

    return value


# ============================================================
# MODEL FEATURE INFORMATION
# ============================================================

def get_expected_features(model):

    """
    Gets feature names from a model/pipeline when available.
    """

    if model is None:

        return None

    try:

        if hasattr(
            model,
            "feature_names_in_"
        ):

            return list(
                model.feature_names_in_
            )

    except Exception:
        pass


    try:

        if hasattr(
            model,
            "named_steps"
        ):

            for _, step in model.named_steps.items():

                if hasattr(
                    step,
                    "feature_names_in_"
                ):

                    return list(
                        step.feature_names_in_
                    )

    except Exception:
        pass


    return None


def get_expected_count(model):

    """
    Gets expected number of input features.
    """

    if model is None:

        return None

    try:

        if hasattr(
            model,
            "n_features_in_"
        ):

            return int(
                model.n_features_in_
            )

    except Exception:
        pass


    try:

        if hasattr(
            model,
            "named_steps"
        ):

            for _, step in model.named_steps.items():

                if hasattr(
                    step,
                    "n_features_in_"
                ):

                    return int(
                        step.n_features_in_
                    )

    except Exception:
        pass


    return None


# ============================================================
# SAFE MODEL PREDICTION
# ============================================================

def run_prediction(model, dataframe):

    """
    Runs prediction while preserving the model's expected
    feature structure.
    """

    if model is None:

        raise RuntimeError(
            "Required disease model was not found "
            "inside the Models folder."
        )


    expected_names = get_expected_features(
        model
    )

    expected_count = get_expected_count(
        model
    )


    # --------------------------------------------------------
    # If model provides feature names
    # --------------------------------------------------------

    if expected_names:

        prepared = pd.DataFrame(
            index=[0]
        )

        for feature in expected_names:

            if feature in dataframe.columns:

                prepared[feature] = dataframe[
                    feature
                ].iloc[0]

            else:

                prepared[feature] = 0


        dataframe = prepared


    # --------------------------------------------------------
    # If model only provides feature count
    # --------------------------------------------------------

    elif expected_count is not None:

        if dataframe.shape[1] < expected_count:

            for i in range(
                dataframe.shape[1],
                expected_count
            ):

                dataframe[
                    f"feature_{i}"
                ] = 0


        elif dataframe.shape[1] > expected_count:

            dataframe = dataframe.iloc[
                :,
                :expected_count
            ]


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        dataframe
    )[0]


    probabilities = None

    if hasattr(
        model,
        "predict_proba"
    ):

        try:

            probabilities = model.predict_proba(
                dataframe
            )[0]

        except Exception:

            probabilities = None


    return prediction, probabilities


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk_score(
    prediction,
    probabilities
):

    if probabilities is not None:

        try:

            probabilities = np.asarray(
                probabilities,
                dtype=float
            )

            probabilities = np.nan_to_num(
                probabilities
            )

            if probabilities.size > 0:

                return round(
                    float(
                        np.max(probabilities)
                    ) * 100,
                    2
                )

        except Exception:

            pass


    try:

        prediction_number = float(
            prediction
        )

        if prediction_number <= 0:

            return 20.0

        return 80.0

    except Exception:

        return 50.0


def get_risk_level(score):

    if score < 30:

        return "Low Risk"

    elif score < 60:

        return "Moderate Risk"

    elif score < 80:

        return "High Risk"

    else:

        return "Very High Risk"


# ============================================================
# PROBABILITY CONVERSION
# ============================================================

def create_probabilities(
    probabilities,
    labels
):

    result = {}

    if probabilities is None:

        return result


    try:

        probabilities = np.asarray(
            probabilities,
            dtype=float
        )

        probabilities = np.nan_to_num(
            probabilities
        )


        for index, probability in enumerate(
            probabilities
        ):

            if index < len(labels):

                result[
                    labels[index]
                ] = round(
                    float(probability) * 100,
                    2
                )

    except Exception:

        pass


    return result


# ============================================================
# DIABETES DATA
# ============================================================

def build_diabetes_data():

    data = {

        "HighBP":
            clean_number(
                get_value(
                    "HighBP",
                    "highbp"
                ),
                0
            ),

        "HighChol":
            clean_number(
                get_value(
                    "HighChol",
                    "highchol"
                ),
                0
            ),

        "CholCheck":
            clean_number(
                get_value(
                    "CholCheck",
                    "cholcheck"
                ),
                1
            ),

        "BMI":
            clean_number(
                get_value("BMI", "bmi"),
                25
            ),

        "Smoker":
            clean_number(
                get_value("Smoker", "smoker"),
                0
            ),

        "Stroke":
            clean_number(
                get_value("Stroke", "stroke"),
                0
            ),

        "HeartDiseaseorAttack":
            clean_number(
                get_value(
                    "HeartDiseaseorAttack",
                    "heartdiseaseorattack",
                    "HeartDisease"
                ),
                0
            ),

        "PhysActivity":
            clean_number(
                get_value(
                    "PhysActivity",
                    "physactivity"
                ),
                1
            ),

        "Fruits":
            clean_number(
                get_value("Fruits", "fruits"),
                1
            ),

        "Veggies":
            clean_number(
                get_value("Veggies", "veggies"),
                1
            ),

        "HvyAlcoholConsump":
            clean_number(
                get_value(
                    "HvyAlcoholConsump",
                    "hvyalcoholconsump"
                ),
                0
            ),

        "AnyHealthcare":
            clean_number(
                get_value(
                    "AnyHealthcare",
                    "anyhealthcare"
                ),
                1
            ),

        "NoDocbcCost":
            clean_number(
                get_value(
                    "NoDocbcCost",
                    "nodocbccost"
                ),
                0
            ),

        "GenHlth":
            clean_number(
                get_value(
                    "GenHlth",
                    "genhlth"
                ),
                3
            ),

        "MentHlth":
            clean_number(
                get_value(
                    "MentHlth",
                    "menthlth"
                ),
                0
            ),

        "PhysHlth":
            clean_number(
                get_value(
                    "PhysHlth",
                    "physhlth"
                ),
                0
            ),

        "DiffWalk":
            clean_number(
                get_value(
                    "DiffWalk",
                    "diffwalk"
                ),
                0
            ),

        "Sex":
            clean_number(
                get_value(
                    "Sex",
                    "sex"
                ),
                0
            ),

        "Age":
            clean_number(
                get_value(
                    "Age",
                    "age"
                ),
                5
            ),

        "Education":
            clean_number(
                get_value(
                    "Education",
                    "education"
                ),
                4
            ),

        "Income":
            clean_number(
                get_value(
                    "Income",
                    "income"
                ),
                5
            )
    }


    # IMPORTANT:
    # This is exactly 21 features.

    columns = [

        "HighBP",
        "HighChol",
        "CholCheck",
        "BMI",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "GenHlth",
        "MentHlth",
        "PhysHlth",
        "DiffWalk",
        "Sex",
        "Age",
        "Education",
        "Income"
    ]


    return pd.DataFrame(
        [[data[column] for column in columns]],
        columns=columns
    )


# ============================================================
# HEART DISEASE DATA
# ============================================================

def build_heart_data():

    data = {

        "age":
            clean_number(
                get_value("age", "Age"),
                50
            ),

        "sex":
            clean_number(
                get_value("sex", "Sex"),
                1
            ),

        "cp":
            clean_number(
                get_value("cp", "CP"),
                0
            ),

        "trestbps":
            clean_number(
                get_value(
                    "trestbps",
                    "Trestbps"
                ),
                120
            ),

        "chol":
            clean_number(
                get_value("chol", "Chol"),
                200
            ),

        "fbs":
            clean_number(
                get_value("fbs", "Fbs"),
                0
            ),

        "restecg":
            clean_number(
                get_value(
                    "restecg",
                    "Restecg"
                ),
                0
            ),

        "thalach":
            clean_number(
                get_value(
                    "thalach",
                    "Thalach"
                ),
                150
            ),

        "exang":
            clean_number(
                get_value("exang", "Exang"),
                0
            ),

        "oldpeak":
            clean_number(
                get_value(
                    "oldpeak",
                    "Oldpeak"
                ),
                0
            ),

        "slope":
            clean_number(
                get_value("slope", "Slope"),
                1
            ),

        "ca":
            clean_number(
                get_value("ca", "CA"),
                0
            ),

        "thal":
            clean_number(
                get_value("thal", "Thal"),
                2
            )
    }


    columns = [

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
        "thal"
    ]


    return pd.DataFrame(
        [[data[column] for column in columns]],
        columns=columns
    )


# ============================================================
# STROKE DATA
# ============================================================

def encode_gender(value):

    value = clean_text(
        value,
        "Unknown"
    ).lower()

    if value == "male":
        return 1

    if value == "female":
        return 0

    return 0


def encode_married(value):

    value = clean_text(
        value,
        "No"
    ).lower()

    return 1 if value == "yes" else 0


def encode_work(value):

    value = clean_text(
        value,
        "Private"
    )

    mapping = {

        "Private": 0,

        "Self-employed": 1,

        "Govt_job": 2,

        "children": 3,

        "Never_worked": 4
    }

    return mapping.get(
        value,
        0
    )


def encode_residence(value):

    value = clean_text(
        value,
        "Urban"
    )

    return 1 if value == "Urban" else 0


def encode_smoking(value):

    value = clean_text(
        value,
        "Unknown"
    )

    mapping = {

        "never smoked": 0,

        "formerly smoked": 1,

        "smokes": 2,

        "Unknown": 3
    }

    return mapping.get(
        value,
        3
    )


def build_stroke_data():

    gender = clean_text(
        get_value(
            "gender",
            "Gender",
            "sex"
        ),
        "Unknown"
    )

    age = clean_number(
        get_value(
            "age",
            "Age"
        ),
        50
    )

    hypertension = clean_number(
        get_value(
            "hypertension",
            "Hypertension"
        ),
        0
    )

    heart_disease = clean_number(
        get_value(
            "heart_disease",
            "heartDisease",
            "HeartDisease"
        ),
        0
    )

    married = clean_text(
        get_value(
            "ever_married",
            "EverMarried"
        ),
        "No"
    )

    work_type = clean_text(
        get_value(
            "work_type",
            "WorkType"
        ),
        "Private"
    )

    residence = clean_text(
        get_value(
            "Residence_type",
            "residence_type",
            "ResidenceType"
        ),
        "Urban"
    )

    glucose = clean_number(
        get_value(
            "avg_glucose_level",
            "AvgGlucoseLevel"
        ),
        100
    )

    bmi = clean_number(
        get_value(
            "bmi",
            "BMI"
        ),
        25
    )

    smoking = clean_text(
        get_value(
            "smoking_status",
            "SmokingStatus"
        ),
        "Unknown"
    )


    # Numeric representation used by common
    # stroke datasets.

    data = {

        "gender":
            encode_gender(gender),

        "age":
            age,

        "hypertension":
            hypertension,

        "heart_disease":
            heart_disease,

        "ever_married":
            encode_married(married),

        "work_type":
            encode_work(work_type),

        "Residence_type":
            encode_residence(residence),

        "avg_glucose_level":
            glucose,

        "bmi":
            bmi,

        "smoking_status":
            encode_smoking(smoking)
    }


    columns = [

        "gender",
        "age",
        "hypertension",
        "heart_disease",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "smoking_status"
    ]


    return pd.DataFrame(
        [[data[column] for column in columns]],
        columns=columns
    )


# ============================================================
# RESULT PAGE
# ============================================================

def render_result(
    prediction,
    probabilities,
    labels,
    model_name,
    disease
):

    score = calculate_risk_score(
        prediction,
        probabilities
    )


    probability_data = create_probabilities(
        probabilities,
        labels
    )


    # If probability is unavailable, still display
    # a meaningful result.

    if not probability_data:

        try:

            if float(prediction) == 0:

                probability_data = {
                    labels[0]: 100.0
                }

            else:

                probability_data = {
                    labels[-1]: 100.0
                }

        except Exception:

            probability_data = {
                "Result": 100.0
            }


    # --------------------------------------------------------
    # DISPLAY PREDICTION
    # --------------------------------------------------------

    try:

        numeric_prediction = float(
            prediction
        )

    except Exception:

        numeric_prediction = None


    if disease == "diabetes":

        if numeric_prediction == 0:

            prediction_text = "No Diabetes"

        elif numeric_prediction == 1:

            prediction_text = "Prediabetes"

        elif numeric_prediction == 2:

            prediction_text = "Diabetes"

        else:

            prediction_text = str(
                prediction
            )


    elif disease == "heart":

        if numeric_prediction == 0:

            prediction_text = (
                "No Heart Disease"
            )

        else:

            prediction_text = (
                "Heart Disease"
            )


    elif disease == "stroke":

        if numeric_prediction == 0:

            prediction_text = "No Stroke"

        else:

            prediction_text = "Stroke Risk"


    else:

        prediction_text = str(
            prediction
        )


    return render_template(

        "result.html",

        prediction=prediction_text,

        risk_score=round(
            score,
            2
        ),

        risk_level=get_risk_level(
            score
        ),

        probabilities=probability_data,

        model_name=model_name,

        used_defaults=[],

        disease=disease
    )


# ============================================================
# HOME
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    if request.method == "POST":

        return perform_prediction()

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    return perform_prediction()


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def perform_prediction():

    try:

        disease = clean_text(
            get_value(
                "disease",
                "disease_type",
                "condition"
            ),
            ""
        ).lower()


        # ----------------------------------------------------
        # AUTO-DETECT DISEASE FROM FORM
        # ----------------------------------------------------

        if not disease:

            form_text = " ".join(
                request.form.keys()
            ).lower()


            if (
                "highbp" in form_text
                or "highchol" in form_text
                or "genhlth" in form_text
            ):

                disease = "diabetes"


            elif (
                "trestbps" in form_text
                or "thalach" in form_text
                or "oldpeak" in form_text
            ):

                disease = "heart"


            elif (
                "hypertension" in form_text
                or "avg_glucose_level" in form_text
                or "smoking_status" in form_text
            ):

                disease = "stroke"


        # ----------------------------------------------------
        # DIABETES
        # ----------------------------------------------------

        if disease in [
            "diabetes",
            "diabetic"
        ]:

            if diabetes_model is None:

                raise RuntimeError(
                    "diabetes_model.pkl was not found "
                    "inside the Models folder."
                )


            X = build_diabetes_data()


            print(
                "[DIABETES] Input shape:",
                X.shape
            )

            print(
                "[DIABETES] Features:",
                list(X.columns)
            )


            prediction, probabilities = run_prediction(
                diabetes_model,
                X
            )


            return render_result(

                prediction,

                probabilities,

                [
                    "No Diabetes",
                    "Prediabetes",
                    "Diabetes"
                ],

                "Diabetes Classification Model",

                "diabetes"
            )


        # ----------------------------------------------------
        # HEART DISEASE
        # ----------------------------------------------------

        if disease in [
            "heart",
            "heart disease",
            "heart_disease",
            "heartdisease"
        ]:

            if heart_model is None:

                raise RuntimeError(
                    "Heart disease model was not found "
                    "inside the Models folder."
                )


            X = build_heart_data()


            print(
                "[HEART] Input shape:",
                X.shape
            )


            prediction, probabilities = run_prediction(
                heart_model,
                X
            )


            return render_result(

                prediction,

                probabilities,

                [
                    "No Heart Disease",
                    "Heart Disease"
                ],

                "Heart Disease Classification Model",

                "heart"
            )


        # ----------------------------------------------------
        # STROKE
        # ----------------------------------------------------

        if disease in [
            "stroke",
            "stroke risk"
        ]:

            if stroke_model is None:

                raise RuntimeError(
                    "Stroke model was not found "
                    "inside the Models folder."
                )


            X = build_stroke_data()


            print(
                "[STROKE] Input shape:",
                X.shape
            )


            prediction, probabilities = run_prediction(
                stroke_model,
                X
            )


            return render_result(

                prediction,

                probabilities,

                [
                    "No Stroke",
                    "Stroke"
                ],

                "Stroke Classification Model",

                "stroke"
            )


        raise RuntimeError(
            "Please select Diabetes, Heart Disease, "
            "or Stroke before calculating risk."
        )


    except Exception as error:

        print()
        print("=" * 65)
        print("PREDICTION ERROR")
        print("=" * 65)
        print(str(error))
        print("=" * 65)

        traceback.print_exc()

        print()


        return render_template(

            "result.html",

            error=(
                "Prediction could not be completed. "
                "Please check the entered information "
                "and model configuration."
            ),

            prediction=None,

            risk_score=0,

            risk_level="Unavailable",

            probabilities={},

            model_name="Healthcare Disease Risk Analysis",

            used_defaults=[]
        )


# ============================================================
# DASHBOARD
# ============================================================

@app.route(
    "/dashboard"
)
def dashboard():

    return render_template(
        "dashboard.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health"
)
def health():

    return {

        "status": "running",

        "models": {

            "diabetes":
                diabetes_model is not None,

            "heart_disease":
                heart_model is not None,

            "stroke":
                stroke_model is not None
        }
    }


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 65)
    print("Starting Healthcare Disease Risk Analysis")
    print("=" * 65)
    print("Open: http://127.0.0.1:5000")
    print("=" * 65)
    print()


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )