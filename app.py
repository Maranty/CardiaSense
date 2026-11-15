import pandas as pd
import joblib
import shap
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="CardiaSense", layout="wide")

ALL_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

ROUTINE_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg']

MODEL_CONFIG = {
    "Linear Regression Model": {
        "file": "lr_model.joblib",
        "features": ALL_FEATURES,
        "shap": True,
    },
    "Linear Regression Model Routine": {
        "file": "routine_lr_model.joblib",
        "features": ROUTINE_FEATURES,
        "shap": True,
    },
    "Baseline Model": {
        "file": "baseline_model.joblib",
        "features": ALL_FEATURES,
        "shap": False,
    },
}


@st.cache_resource
def load_assets():
    models = {
        name: joblib.load(cfg["file"]) for name, cfg in MODEL_CONFIG.items()
    }
    shap_background = joblib.load("shap_background.joblib")
    return models, shap_background


models, shap_background = load_assets()


def get_background(features):

    bg_df = pd.DataFrame(shap_background, columns=ALL_FEATURES)
    return bg_df[features]


@st.cache_resource
def get_explainer(model_name):
    cfg = MODEL_CONFIG[model_name]
    model = models[model_name]
    features = cfg["features"]
    disease_idx = list(model.classes_).index(1)

    def predict_fn(data):
        df = pd.DataFrame(data, columns=features)
        return model.predict_proba(df)[:, disease_idx]

    return shap.Explainer(predict_fn, get_background(features))


@st.cache_data
def calculate_individual_shap(model_name, user_input_tuple):
    features = MODEL_CONFIG[model_name]["features"]
    user_input_df = pd.DataFrame([user_input_tuple], columns=features)
    return get_explainer(model_name)(user_input_df)


@st.cache_data
def calculate_global_shap(model_name):
    features = MODEL_CONFIG[model_name]["features"]
    return get_explainer(model_name)(get_background(features))


def heart_disease_risk_prediction(model_name, user_input):
    cfg = MODEL_CONFIG[model_name]
    model = models[model_name]
    features = cfg["features"]
    disease_index = list(model.classes_).index(1)

    user_input_df = pd.DataFrame([user_input], columns=features)
    prediction = model.predict(user_input_df)[0]
    proba = model.predict_proba(user_input_df)[0]

    disease_prob = proba[disease_index]
    no_disease_prob = proba[1 - disease_index]

    status_color = "#ff4b4b" if prediction == 1 else "#21c354"
    status_text = "Heart Disease" if prediction == 1 else "No Heart Disease"

    st.markdown(
        f"""
        <div class="prediction-panel">
            <h3>Prediction Results</h3>
            <p style="opacity:0.7; font-size:13px;">Model: {model_name}</p>
            <p>Heart Disease Probability</p>
            <h2>{disease_prob:.2%}</h2>
            <p>No Heart Disease Probability</p>
            <h2>{no_disease_prob:.2%}</h2>
            <hr>
            <h4>Model Prediction</h4>
            <h3 style="color: {status_color};">{status_text}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not cfg["shap"]:
        st.info("SHAP explanations are not available for the Baseline Model.")
        return

    shap_values = calculate_individual_shap(model_name, tuple(user_input))
    st.subheader("Individual Prediction Explanation")
    st.write(
        "The SHAP waterfall plot shows how each input feature pushes this specific patient's risk higher or lower."
    )
    shap.plots.waterfall(shap_values[0], show=False, max_display=14)
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    st.pyplot(fig, width='stretch')
    plt.close(fig)

    global_shap_values = calculate_global_shap(model_name)
    st.subheader("Global Feature Importance")
    st.write(
        "The SHAP beeswarm plot shows how features impact predictions across the broader reference dataset."
    )
    shap.plots.beeswarm(global_shap_values, show=False, max_display=14)
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    st.pyplot(fig, width='stretch')
    plt.close(fig)


def main():
    st.title("Heart Disease Risk Predictor")

    st.markdown("""
    <style>

    /* Main page layout */
    .stMainBlockContainer {
        max-width: 1400px;
        margin-left: auto;
        margin-right: auto;
        padding-left: 2rem;
        padding-right: 340px;
        box-sizing: border-box;
    }

    /* Prediction panel */
    .prediction-panel {
        position: fixed;
        top: 80px;
        right: 25px;
        width: 290px;
        z-index: 999999;
        display: block;
        visibility: visible;
        opacity: 1;

        background-color: var(--secondary-background-color);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.4);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    /* Panel text */
    .prediction-panel h3 {
        margin-top: 0;
        font-size: 20px;
    }

    .prediction-panel p {
        margin-bottom: 8px;
    }

    /* Smaller screens */
    @media (max-width: 900px) {

        .stMainBlockContainer {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .prediction-panel {
            position: relative;
            top: auto;
            right: auto;
            width: 100%;
            margin-top: 20px;
            margin-bottom: 20px;
        }
    }

    </style>
    """, unsafe_allow_html=True)

    # Model selector
    model_name = st.radio(
        "Select Model:",
        options=list(MODEL_CONFIG.keys()),
        index=0,
        horizontal=True,
        help="The Routine model uses only basic information and routine clinical measurements.",
    )
    features = MODEL_CONFIG[model_name]["features"]

    def show(f):
        return f in features

    values = {}

    # 1. Basic Information
    st.subheader("1. Basic Information")

    values['age'] = st.slider(
        label="Age:",
        min_value=1,
        max_value=100,
        value=50,
        help="Enter the patient's age in years."
    )

    values['sex'] = st.radio(
        "Sex:",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male",
        horizontal=True,
        help="Select the patient's biological sex."
    )

    st.divider()

    # 2. Clinical Measurements
    st.subheader("2. Clinical Measurements")

    values['trestbps'] = st.number_input(
        label="Resting Blood Pressure (mm Hg):",
        min_value=50,
        max_value=250,
        value=120,
        step=1,
        help="Enter resting blood pressure in mm Hg on admission to the hospital."
    )

    values['chol'] = st.number_input(
        label="Serum Cholesterol (mg/dl):",
        min_value=50,
        max_value=600,
        value=200,
        step=1,
        help="Enter total serum cholesterol in mg/dl."
    )

    values['fbs'] = st.selectbox(
        label="Fasting Blood Sugar > 120 mg/dl:",
        options=[0, 1],
        format_func=lambda x: "No (≤ 120 mg/dl)" if x == 0 else "Yes (> 120 mg/dl)",
        help="Select whether fasting blood sugar exceeds 120 mg/dl."
    )

    if show('thalach'):
        values['thalach'] = st.number_input(
            label="Maximum Heart Rate Achieved (bpm):",
            min_value=60,
            max_value=220,
            value=150,
            step=1,
            help="Enter the maximum heart rate achieved during exercise testing."
        )

    st.divider()

    # 3. ECG & Exercise Test Results
    st.subheader("3. ECG & Exercise Test Results" if show('exang') else "3. ECG Results")

    cp_options = {
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-anginal Pain",
        4: "Asymptomatic"
    }

    values['cp'] = st.selectbox(
        label="Chest Pain Type:",
        options=list(cp_options.keys()),
        format_func=lambda x: cp_options[x],
        help="Select the type of chest pain experienced."
    )

    restecg_options = {
        0: "Normal",
        1: "ST-T wave abnormality (T wave inversions and/or ST elevation or depression > 0.05 mV)",
        2: "Probable or definite left ventricular hypertrophy by Estes' criteria"
    }

    values['restecg'] = st.selectbox(
        label="Resting Electrocardiographic Results:",
        options=list(restecg_options.keys()),
        format_func=lambda x: restecg_options[x],
        help="Select the patient's resting ECG result."
    )

    if show('exang'):
        values['exang'] = st.selectbox(
            label="Exercise-Induced Angina:",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Select whether exercise causes angina."
        )

    if show('oldpeak'):
        values['oldpeak'] = st.number_input(
            label="ST Depression Induced by Exercise Relative to Rest:",
            min_value=0.0,
            max_value=6.5,
            value=1.0,
            step=0.1,
            format="%.1f",
            help="Enter the ST depression measured during exercise (mm)."
        )

    if show('slope'):
        slope_options = {
            1: "Upsloping",
            2: "Flat",
            3: "Downsloping"
        }

        values['slope'] = st.selectbox(
            label="Slope of the Peak Exercise ST Segment:",
            options=list(slope_options.keys()),
            format_func=lambda x: slope_options[x],
            index=1,
            help="Select the ST segment slope during peak exercise."
        )

    st.divider()

    # 4. Additional Clinical Findings (hidden entirely for the Routine model)
    if show('ca') or show('thal'):
        st.subheader("4. Additional Clinical Findings")

        if show('ca'):
            values['ca'] = st.slider(
                label="Number of Major Vessels Colored by Fluoroscopy:",
                min_value=0,
                max_value=3,
                value=0,
                step=1,
                help="Select the number of major vessels visible by fluoroscopy (0–3)."
            )

        if show('thal'):
            thal_options = {
                3: "Normal",
                6: "Fixed Defect",
                7: "Reversible Defect"
            }

            values['thal'] = st.selectbox(
                label="Thalassemia:",
                options=[3, 6, 7],
                format_func=lambda x: thal_options[x],
                help="Select the thallium stress test result."
            )

        st.divider()

    # Real-time prediction (inputs ordered to match the selected model's features)
    user_input = [values[f] for f in features]
    heart_disease_risk_prediction(model_name, user_input)

    # Disclaimer
    st.warning(
        "⚠️ **Disclaimer:** For educational and hackathon demonstration purposes only. "
        "Not a substitute for professional medical advice or diagnosis."
    )


if __name__ == "__main__":
    main()
