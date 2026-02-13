"""
Streamlit Web Application for CKD Prediction System

Uses kidney_disease.csv schema:
- Dynamic input fields based on dataset columns
- No hardcoded features
- No synthetic/demo fallback
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import joblib

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from data_pipeline import CKDDataPipeline
from ai_assistant import CKDAssistant

st.set_page_config(
    page_title="CKD Risk Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Default ranges for numerical columns (user-friendly names)
NUMERICAL_DEFAULTS = {
    "age": (1, 120, 50),
    "blood_pressure": (50, 200, 80),
    "specific_gravity": (1.0, 1.05, 1.02),
    "albumin": (0, 5, 0),
    "sugar": (0, 5, 0),
    "blood_glucose_random": (50, 500, 121),
    "blood_urea": (10, 400, 36),
    "serum_creatinine": (0.5, 15, 1.2),
    "sodium": (100, 180, 140),
    "potassium": (2, 10, 4),
    "haemoglobin": (3, 20, 12),
    "packed_cell_volume": (20, 60, 40),
    "white_blood_cell_count": (2000, 20000, 8000),
    "red_blood_cell_count": (2, 8, 5),
}


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    
    .main-header { 
        font-family: 'DM Sans', sans-serif;
        font-size: 2.5rem; font-weight: 700; 
        background: linear-gradient(135deg, #0D47A1 0%, #1976D2 50%, #42A5F5 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        text-align: center; margin-bottom: 2rem; letter-spacing: -0.5px;
    }
    .sub-header { font-size: 1.5rem; color: #37474F; margin-bottom: 1rem; font-weight: 600; }
    
    .risk-low { background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%); color: #1B5E20; 
        padding: 1.5rem; border-radius: 16px; text-align: center; font-weight: 600; 
        box-shadow: 0 4px 20px rgba(76,175,80,0.2); border: 1px solid rgba(76,175,80,0.3); }
    .risk-medium { background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%); color: #E65100; 
        padding: 1.5rem; border-radius: 16px; text-align: center; font-weight: 600; 
        box-shadow: 0 4px 20px rgba(255,152,0,0.2); border: 1px solid rgba(255,152,0,0.3); }
    .risk-high { background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%); color: #B71C1C; 
        padding: 1.5rem; border-radius: 16px; text-align: center; font-weight: 600; 
        box-shadow: 0 4px 20px rgba(244,67,54,0.2); border: 1px solid rgba(244,67,54,0.3); }
    
    .metric-card { background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%); 
        padding: 1.6rem; border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); 
        border-left: 5px solid #1976D2; color: #263238; margin-bottom: 1rem; 
        transition: transform 0.2s ease, box-shadow 0.2s ease; }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 12px 28px rgba(0,0,0,0.12); }
    .metric-card h3 { color: #0D47A1; font-weight: 700; margin-bottom: 0.5rem; }
    .metric-card p { color: #546E7A; font-size: 0.95rem; line-height: 1.6; }
    
    .info-box { background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%); 
        padding: 1.2rem 1.5rem; border-radius: 12px; border-left: 5px solid #1976D2; 
        color: #0D47A1; font-weight: 500; box-shadow: 0 2px 12px rgba(25,118,210,0.15); }
    
    .rec-card { background: linear-gradient(145deg, #ffffff 0%, #f1f8e9 100%); 
        padding: 1.2rem 1.5rem; border-radius: 12px; margin-bottom: 1rem; 
        box-shadow: 0 4px 16px rgba(0,0,0,0.06); border-left: 4px solid #7CB342;
        transition: all 0.25s ease; font-size: 1.05rem; line-height: 1.6; color: #263238; }
    .rec-card:hover { box-shadow: 0 6px 20px rgba(124,179,66,0.25); transform: translateX(4px); }
    
    .prevention-card { background: linear-gradient(145deg, #ffffff 0%, #fff8e1 100%); 
        padding: 1.4rem; border-radius: 16px; margin: 0.8rem 0; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.06); text-align: center;
        border: 1px solid rgba(255,193,7,0.3); transition: all 0.25s ease; }
    .prevention-card:hover { box-shadow: 0 8px 28px rgba(255,193,7,0.2); transform: translateY(-3px); }
    .prevention-card h4 { color: #F57F17; font-weight: 700; margin-bottom: 0.5rem; font-size: 1.1rem; }
    .prevention-card p { color: #5D4037; font-size: 0.9rem; line-height: 1.5; margin: 0; }
    
    .stage-row { padding: 0.8rem 1rem; margin: 0.4rem 0; border-radius: 10px; 
        background: linear-gradient(90deg, #E3F2FD 0%, transparent 100%); 
        border-left: 4px solid #1976D2; transition: background 0.2s; }
    .stage-row:hover { background: linear-gradient(90deg, #BBDEFB 0%, transparent 100%); }
    
    .sidebar .stRadio > label { font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    models_path = Path(__file__).parent / "models"

    try:
        # If models folder missing → train models automatically
        if not models_path.exists():
            import subprocess
            import sys
        subprocess.run([sys.executable, "src/train.py"])


        pipeline = CKDDataPipeline()
        pipeline.load_pipeline(str(models_path))

        classifier = joblib.load(str(models_path / "best_classifier.pkl"))
        regressor = joblib.load(str(models_path / "best_regressor.pkl"))
        model_info = joblib.load(str(models_path / "model_info.pkl"))

        return pipeline, classifier, regressor, model_info

    except Exception as e:
        st.warning("Models not found. Running in demo mode.")
        return None, None, None, None



def create_gauge_chart(value, title, max_value=100):
    """Create a gauge chart for CKD risk."""
    if value < 30:
        color = "#4CAF50"
    elif value < 70:
        color = "#FFC107"
    else:
        color = "#F44336"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 20}},
            number={"suffix": "%", "font": {"size": 40}},
            gauge={
                "axis": {"range": [0, max_value], "tickwidth": 1},
                "bar": {"color": color},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, max_value * 0.3], "color": "#E8F5E9"},
                    {"range": [max_value * 0.3, max_value * 0.7], "color": "#FFF8E1"},
                    {"range": [max_value * 0.7, max_value], "color": "#FFEBEE"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": value},
            },
        )
    )
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_risk_factors_chart(patient_data, feature_names, feature_importance=None):
    """Create a bar chart of top risk factors (if feature importance available)."""
    if feature_importance is None or len(feature_importance) == 0:
        return None
    df = pd.DataFrame(feature_importance).head(10)
    df["feature"] = df["feature"].apply(lambda x: str(x).replace("_", " ").title())
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="RdYlGn_r",
    )
    fig.update_layout(
        title="Top 10 Risk Factors",
        xaxis_title="Importance",
        yaxis_title="",
        height=400,
        showlegend=False,
        yaxis={"categoryorder": "total ascending"},
    )
    return fig


def build_input_form(schema, pipeline):
    """Dynamically build input form from schema. Returns patient_data dict."""
    numerical = schema.get("numerical", [])
    categorical = schema.get("categorical", [])
    cat_options = schema.get("categorical_options", {})

    patient_data = {}

    # Numerical inputs
    if numerical:
        st.markdown("##### 📊 Numerical / Lab Values")
        cols = st.columns(min(4, len(numerical)))
        for i, col_name in enumerate(numerical):
            with cols[i % len(cols)]:
                min_v, max_v, default = NUMERICAL_DEFAULTS.get(
                    col_name, (0, 1000, 0)
                )
                step_val = 1 if col_name == "age" else (0.1 if col_name in ("specific_gravity", "serum_creatinine", "haemoglobin", "potassium", "sodium") else 1.0)
                val = st.number_input(
                    col_name.replace("_", " ").title(),
                    min_value=int(min_v) if col_name == "age" else float(min_v),
                    max_value=int(max_v) if col_name == "age" else float(max_v),
                    value=int(default) if col_name == "age" else float(default),
                    step=step_val,
                    format="%d" if col_name == "age" else None,
                    key=f"num_{col_name}",
                )
                patient_data[col_name] = int(val) if col_name == "age" else val

    # Categorical inputs
    if categorical:
        st.markdown("##### 🩺 Categorical / Clinical")
        cols = st.columns(min(4, len(categorical)))
        for i, col_name in enumerate(categorical):
            with cols[i % len(cols)]:
                options = cat_options.get(col_name, ["yes", "no", "normal", "abnormal", "good", "poor", "present", "notpresent"])
                if not options:
                    options = ["yes", "no"]
                display_name = col_name.replace("_", " ").title()
                val = st.selectbox(display_name, options=options, key=f"cat_{col_name}")
                patient_data[col_name] = val

    # Lifestyle & Family History (used for recommendations only; not in kidney_disease.csv)
    st.markdown("##### 🏠 Lifestyle & Family History")
    st.caption("Used for personalized recommendations only")
    lc1, lc2 = st.columns(2)
    with lc1:
        smoking = st.selectbox("Smoking", options=["No", "Yes"], key="lifestyle_smoking", help="Do you currently smoke?")
        patient_data["smoking"] = smoking
    with lc2:
        family_ckd = st.selectbox("Family History of CKD", options=["No", "Yes"], key="lifestyle_family_ckd", help="Has anyone in your family had kidney disease?")
        patient_data["family_history_ckd"] = family_ckd

    return patient_data


def main():
    st.markdown('<h1 class="main-header">🏥 Chronic Kidney Disease Risk Prediction System</h1>', unsafe_allow_html=True)
    assistant = CKDAssistant()

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/kidney.png", width=80)
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "",
            ["🔮 Prediction", "📊 About CKD", "🧠 How It Works", "📈 Model Performance"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Chronic Kidney Disease Risk Prediction")

    with st.spinner("Loading AI models..."):
        pipeline, classifier, regressor, model_info = load_models()

    models_available = pipeline is not None and classifier is not None

    if page == "🔮 Prediction":
        st.markdown("### Enter Patient Information")

        if not models_available:
            st.warning("⚠️ Models not found. Please run training first: `python src/train.py`")
            st.stop()

        schema = pipeline.get_feature_schema()
        patient_data = build_input_form(schema, pipeline)

        st.markdown("---")
        if st.button("🔍 Predict CKD Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing patient data..."):
                try:
                    X = pipeline.preprocess_single_patient(patient_data)
                    ckd_prob = float(classifier.predict_proba(X)[0][1])
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.stop()

            kidney_score = None
            if regressor is not None:
                try:
                    kidney_score = float(regressor.predict(X)[0])
                except Exception:
                    pass

            mapped = dict(patient_data)
            mapped.setdefault("blood_pressure_systolic", mapped.get("blood_pressure"))
            mapped.setdefault("blood_glucose", mapped.get("blood_glucose_random"))
            mapped.setdefault("hemoglobin", mapped.get("haemoglobin"))
            mapped.setdefault("smoking", mapped.get("smoking", "No"))
            mapped.setdefault("family_history_ckd", mapped.get("family_history_ckd", "No"))
            recommendations = assistant.get_lifestyle_recommendations(mapped)

            st.session_state["prediction_results"] = {
                "ckd_prob": ckd_prob,
                "kidney_score": kidney_score,
                "patient_data": patient_data,
                "mapped": mapped,
                "recommendations": recommendations,
            }

        if "prediction_results" in st.session_state:
            res = st.session_state["prediction_results"]
            ckd_prob = res["ckd_prob"]
            kidney_score = res["kidney_score"]
            patient_data = res["patient_data"]
            mapped = res["mapped"]
            recommendations = res["recommendations"]

            st.markdown("---")
            st.markdown("## 📊 Prediction Results")

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.plotly_chart(create_gauge_chart(ckd_prob * 100, "CKD Risk"), use_container_width=True)

            with col_res2:
                if kidney_score is not None:
                    st.plotly_chart(
                        create_gauge_chart(kidney_score, "Kidney Function Score", max_value=100),
                        use_container_width=True,
                    )
                else:
                    risk_interp = assistant.interpret_risk_level(ckd_prob)
                    st.markdown(
                        f'<div class="metric-card"><h3>Risk Level: {risk_interp["level"]}</h3>'
                        f'<p>{risk_interp["description"]}</p></div>',
                        unsafe_allow_html=True,
                    )

            risk_interp = assistant.interpret_risk_level(ckd_prob)
            risk_class = f"risk-{risk_interp['level'].lower()}"
            st.markdown(
                f'<div class="{risk_class}"><h2>Risk Level: {risk_interp["level"]}</h2>'
                f'<p style="font-size: 1.2rem;">CKD Probability: {ckd_prob*100:.1f}%</p></div>',
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("#### Feature Importance")
                try:
                    if hasattr(classifier, "feature_importances_"):
                        imp = classifier.feature_importances_
                    else:
                        imp = getattr(classifier, "coef_", None)
                        if imp is not None:
                            imp = np.abs(imp[0])
                    if imp is not None and schema.get("feature_names"):
                        fi = [
                            {"feature": schema["feature_names"][i], "importance": imp[i]}
                            for i in range(min(len(imp), len(schema["feature_names"])))
                        ]
                        fig = create_risk_factors_chart(patient_data, schema["feature_names"], fi)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

            with col_chart2:
                if kidney_score is not None:
                    kidney_interp = assistant.interpret_kidney_function_score(kidney_score)
                    st.markdown(
                        f'<div class="metric-card"><h3>{kidney_interp["status"]}</h3>'
                        f'<p><strong>Stage:</strong> {kidney_interp["stage"]}</p>'
                        f'<p>{kidney_interp["description"]}</p></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("### 📋 Personalized Recommendations")
            st.markdown("*Based on your lab values, conditions, smoking status, and family history*")

            def _categorize(r):
                rl = r.lower()
                if "exercise" in rl or "weight" in rl or "smoking" in rl or "physical" in rl or "active" in rl or "family" in rl:
                    return "Lifestyle"
                if "blood" in rl or "doctor" in rl or "monitor" in rl or "hypertension" in rl or "diabetes" in rl or "provider" in rl:
                    return "Medical"
                return "Diet & Hydration"

            rec_filter = st.selectbox("Filter recommendations by category", ["All", "Lifestyle", "Medical", "Diet & Hydration"], key="rec_filter")
            filtered = [r for r in recommendations if rec_filter == "All" or _categorize(r) == rec_filter]
            if not filtered:
                filtered = recommendations

            rec_icons = {"Lifestyle": "🏃", "Medical": "🩺", "Diet & Hydration": "🥗"}
            cols = st.columns(2)
            for i, rec in enumerate(filtered):
                cat = _categorize(rec)
                icon = rec_icons.get(cat, "✅")
                with cols[i % 2]:
                    st.markdown(
                        f'<div class="rec-card">'
                        f'<span style="font-size:1.4rem;">{icon}</span> '
                        f'<strong style="font-size:1.05rem;">{rec}</strong></div>',
                        unsafe_allow_html=True,
                    )

            with st.expander("📄 View Detailed Report"):
                report = assistant.generate_patient_report(
                    mapped, ckd_prob, kidney_score if kidney_score is not None else 0
                )
                st.text(report)

    elif page == "📊 About CKD":
        st.markdown("## Understanding Chronic Kidney Disease")
        tabs = st.tabs(["📖 Overview", "📊 Stages", "⚠️ Risk Factors", "🛡️ Prevention & Lifestyle"])
        with tabs[0]:
            st.markdown("""
            ### What is Chronic Kidney Disease?
            Chronic Kidney Disease (CKD) is a long-term condition characterized by gradual loss of kidney function.
            The kidneys filter waste and excess fluids from blood—when damaged, dangerous levels can build up.
            Early detection can slow or prevent kidney failure.
            """)
            st.info("💡 **Key fact:** About 10% of the global population has CKD. Many don't know until it's advanced.")
        with tabs[1]:
            stages_data = [
                ("Stage 1", "≥90", "Normal kidney function", "Monitor & control risk factors"),
                ("Stage 2", "60-89", "Mildly decreased", "Regular monitoring"),
                ("Stage 3a", "45-59", "Mild to moderate", "More frequent checks"),
                ("Stage 3b", "30-44", "Moderate to severe", "Specialist referral"),
                ("Stage 4", "15-29", "Severely decreased", "Prepare for treatment"),
                ("Stage 5", "<15", "Kidney failure", "Dialysis or transplant"),
            ]
            for stage, gfr, desc, action in stages_data:
                st.markdown(
                    f'<div class="stage-row">'
                    f'<strong>{stage}</strong> | GFR: {gfr} | {desc}<br>'
                    f'<small style="color:#546E7A;">→ {action}</small></div>',
                    unsafe_allow_html=True,
                )
        with tabs[2]:
            risk_filter = st.selectbox("Show factors by type", ["All", "Modifiable", "Non-modifiable"], key="risk_filter")
            modifiable = [
                ("🩺", "Diabetes", "Uncontrolled blood sugar damages kidneys over time"),
                ("💓", "Hypertension", "High BP strains blood vessels in kidneys"),
                ("🚬", "Smoking", "Reduces blood flow and accelerates damage"),
                ("🍔", "Obesity", "Increases diabetes and BP risk"),
                ("💊", "Medications", "Overuse of NSAIDs can harm kidneys"),
            ]
            non_modifiable = [
                ("👴", "Age 60+", "Kidney function naturally declines"),
                ("👪", "Family history", "Genetic predisposition to kidney disease"),
                ("🧬", "Ethnicity", "Higher rates in some populations"),
                ("🏥", "Past AKI", "History of acute kidney injury"),
            ]
            items = modifiable if risk_filter == "Modifiable" else (non_modifiable if risk_filter == "Non-modifiable" else modifiable + non_modifiable)
            for icon, title, desc in items:
                st.markdown(
                    f'<div class="rec-card">'
                    f'<span style="font-size:1.3rem;">{icon}</span> <strong>{title}</strong><br>'
                    f'<small style="color:#546E7A;">{desc}</small></div>',
                    unsafe_allow_html=True,
                )
        with tabs[3]:
            st.markdown("#### How to protect your kidneys")
            prev_tips = [
                ("🩺", "Control Blood Sugar", "Keep diabetes under control—high glucose damages kidney filters."),
                ("💓", "Manage Blood Pressure", "Aim for <130/80 mmHg. Check regularly."),
                ("🥗", "Eat Kidney-Friendly", "Low sodium, moderate protein, limit processed foods."),
                ("🏃", "Stay Active", "150 min/week of moderate exercise."),
                ("🚭", "Quit Smoking", "Smoking reduces blood flow to kidneys."),
                ("💧", "Stay Hydrated", "Drink adequate water—avoid sugary drinks."),
                ("💊", "Limit NSAIDs", "Avoid overusing ibuprofen, naproxen."),
                ("🏥", "Get Checked", "Annual kidney function test (creatinine, GFR)."),
            ]
            p_cols = st.columns(4)
            for i, (icon, title, desc) in enumerate(prev_tips):
                with p_cols[i % 4]:
                    st.markdown(
                        f'<div class="prevention-card">'
                        f'<div style="font-size:2rem;margin-bottom:0.5rem;">{icon}</div>'
                        f'<h4>{title}</h4><p>{desc}</p></div>',
                        unsafe_allow_html=True,
                    )

    elif page == "🧠 How It Works":
        st.markdown("## How Our Prediction System Works")
        st.markdown(
            '<div class="info-box">Our system uses Machine Learning (Random Forest, XGBoost, Logistic Regression) '
            'trained on real patient data to predict CKD risk.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        step = st.radio("Explore the pipeline", ["1️⃣ Data Input", "2️⃣ Preprocessing", "3️⃣ ML Models", "4️⃣ Prediction"], horizontal=True, label_visibility="collapsed")
        if "1" in step:
            st.markdown("#### 1️⃣ Data Input")
            st.markdown("We collect lab values and clinical features: age, blood pressure, creatinine, hemoglobin, urine analysis, and medical history.")
            st.code("Features: numerical (age, BP, creatinine...) + categorical (hypertension, diabetes...)")
        elif "2" in step:
            st.markdown("#### 2️⃣ Preprocessing")
            st.markdown("Data is cleaned, missing values imputed (median/mode), categoricals one-hot encoded, and numericals scaled.")
            st.code("Missing values → Impute | Categorical → OneHotEncoder | Numerical → StandardScaler")
        elif "3" in step:
            st.markdown("#### 3️⃣ ML Models")
            st.markdown("Multiple classifiers are trained and the best (by ROC-AUC) is selected for predictions.")
            st.code("Random Forest | XGBoost | Logistic Regression → Best model chosen")
        else:
            st.markdown("#### 4️⃣ Prediction")
            st.markdown("Your input is preprocessed and fed to the model. Output: CKD probability (0–100%).")
            st.code("Input → Preprocess → Model → Probability")

    elif page == "📈 Model Performance":
        st.markdown("## Model Performance")
        if model_info:
            cls_metrics = model_info.get("classification_metrics", {})
            if cls_metrics:
                df_metrics = pd.DataFrame(cls_metrics)
                # Convert to percentage if values are 0-1 (backwards compatibility)
                for col in df_metrics.columns:
                    if col != "Model" and df_metrics[col].dtype in (float, int) and df_metrics[col].max() <= 1:
                        df_metrics[col] = (df_metrics[col] * 100).round(2)
                st.dataframe(df_metrics, use_container_width=True)
            st.markdown(f"**Best Classifier**: {model_info.get('best_classifier', 'N/A').replace('_', ' ').title()}")
            if model_info.get("has_regression"):
                st.markdown(f"**Best Regressor**: {model_info.get('best_regressor', 'N/A').replace('_', ' ').title()}")
        else:
            st.info("Run training first: `python src/train.py`")

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#90A4AE;font-size:0.85rem;padding:1rem;">'
        'CKD Risk Prediction System · For educational use only · Consult a healthcare professional for medical advice</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
