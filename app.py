import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Page Configuration
st.set_page_config(
    page_title="EndoCare AI | Clinical Diabetes Risk Portal",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS Injection for Clean UI)
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.05);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 14px;
        border-radius: 10px;
    }
    /* Button refinement */
    div.stButton > button:first-child {
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# Model Training & Data Caching
@st.cache_resource
def prepare_engine():
    data = pd.read_csv("dataset.csv")
    
    # Feature columns
    feature_cols = [
        'Pregnancies', 'Glucose', 'BloodPressure', 
        'SkinThickness', 'Insulin', 'BMI', 
        'DiabetesPedigreeFunction', 'Age'
    ]
    X = data[feature_cols]
    y = data['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return data, model, acc, cm

try:
    df, model, accuracy, cm = prepare_engine()
except Exception as err:
    st.error(f"Failed to load dataset: {err}")
    st.stop()

# Header Section
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🩺 EndoCare AI Diagnostic Portal")
    st.caption("Clinical Decision Support Tool for Type-2 Diabetes Risk Stratification")
with header_col2:
    st.metric("Model Reliability", f"{accuracy * 100:.1f}%", help="Cross-validated Random Forest accuracy on clinical holdout set.")

st.divider()

# Top-level Tabs
tab_assessment, tab_analytics, tab_clinical_guide = st.tabs([
    "📋 Patient Risk Assessment",
    "📊 Dataset & Model Insights",
    "📖 Clinical Reference Standards"
])

# TAB 1: Assessment Interface
with tab_assessment:
    st.markdown("##### Enter Patient Physiological Parameters")
    
    with st.form("risk_assessment_form"):
        col_vital1, col_vital2 = st.columns(2)
        
        with col_vital1:
            st.markdown("**Metabolic & Glycemic Profile**")
            glucose = st.number_input(
                "Fasting Plasma Glucose (mg/dL)", 
                min_value=40.0, max_value=300.0, value=115.0, step=1.0,
                help="Normal fasting range: 70–99 mg/dL. Prediabetes: 100–125 mg/dL."
            )
            insulin = st.number_input(
                "2-Hour Serum Insulin (mu U/ml)", 
                min_value=0.0, max_value=850.0, value=79.0, step=1.0,
                help="Normal range varies: 16–166 mu U/ml."
            )
            bmi = st.number_input(
                "Body Mass Index (BMI in kg/m²)", 
                min_value=10.0, max_value=70.0, value=28.5, step=0.1,
                help="Healthy weight: 18.5–24.9, Overweight: 25.0–29.9, Obese: ≥30.0."
            )
            skin_thickness = st.number_input(
                "Triceps Skin Fold Thickness (mm)", 
                min_value=0.0, max_value=100.0, value=23.0, step=1.0,
                help="Subcutaneous adipose tissue measurement."
            )

        with col_vital2:
            st.markdown("**Demographic & Cardiovascular Profile**")
            age = st.number_input(
                "Age (Years)", 
                min_value=18, max_value=110, value=38, step=1,
                help="T2D incidence increases markedly past age 35."
            )
            bp = st.number_input(
                "Diastolic Blood Pressure (mm Hg)", 
                min_value=30.0, max_value=150.0, value=72.0, step=1.0,
                help="Optimal: <80 mm Hg. Stage 1 hypertension: 80–89 mm Hg."
            )
            pregnancies = st.number_input(
                "Number of Pregnancies", 
                min_value=0, max_value=20, value=1, step=1,
                help="Includes history of gestational diabetes triggers."
            )
            pedigree = st.number_input(
                "Diabetes Pedigree Function", 
                min_value=0.05, max_value=2.50, value=0.45, step=0.01,
                help="Genetic influence score derived from family medical history."
            )
        
        submit_btn = st.form_submit_button("⚡ Run Stratification Analysis", use_container_width=True, type="primary")

    if submit_btn:
        patient_vector = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, pedigree, age]])
        prediction = model.predict(patient_vector)[0]
        probabilities = model.predict_proba(patient_vector)[0]
        risk_percentage = probabilities[1] * 100

        st.subheader("Diagnostic Risk Stratification Result")
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            # Clinical Gauge Meter
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_percentage,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "T2D Risk Probability (%)", 'font': {'size': 18}},
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "#1E88E5"},
                    'steps': [
                        {'range': [0, 35], 'color': "rgba(46, 204, 113, 0.3)"},
                        {'range': [35, 65], 'color': "rgba(241, 196, 15, 0.3)"},
                        {'range': [65, 100], 'color': "rgba(231, 76, 60, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_percentage
                    }
                }
            ))
            gauge_fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_res2:
            st.write("### Clinical Evaluation")
            if prediction == 1:
                st.error("🔴 **Elevated Risk Detected**")
                st.write(
                    f"The statistical model indicates a **high probability ({risk_percentage:.1f}%)** of diabetes or prediabetic condition."
                )
                st.markdown("""
                *Recommendation:*
                - Order formal Glycated Hemoglobin (HbA1c) diagnostic verification.
                - Refer patient for dietary lifestyle consultation and metabolic monitoring.
                """)
            else:
                st.success("🟢 **Low Risk Profile**")
                st.write(
                    f"The statistical model indicates a **low probability ({risk_percentage:.1f}%)** of diabetes."
                )
                st.markdown("""
                *Recommendation:*
                - Maintain routine check-ups.
                - Encourage standard physical activity and balanced nutritional guidelines.
                """)

# TAB 2: Model & Dataset Insights
with tab_analytics:
    st.subheader("Cohort Distribution & Validation Performance")
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.markdown("**Cohort Outcome Split**")
        outcome_counts = df['Outcome'].value_counts().rename({0: 'Negative (0)', 1: 'Diabetic (1)'})
        pie_fig = go.Figure(data=[go.Pie(labels=outcome_counts.index, values=outcome_counts.values, hole=.4)])
        pie_fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(pie_fig, use_container_width=True)

    with col_chart2:
        st.markdown("**Holdout Validation Matrix**")
        st.dataframe(
            pd.DataFrame(
                cm,
                columns=['Predicted: Non-Diabetic', 'Predicted: Diabetic'],
                index=['Actual: Non-Diabetic', 'Actual: Diabetic']
            ),
            use_container_width=True
        )
        st.caption(f"Evaluated on 20% holdout validation records (Total: {len(df)} entries).")

    with st.expander("Explore Raw Dataset Sample"):
        st.dataframe(df.head(10), use_container_width=True)

# TAB 3: Reference Guidelines
with tab_clinical_guide:
    st.subheader("Diagnostic Criteria & Threshold References")
    
    ref_table = {
        "Biomarker": ["Fasting Glucose", "Diastolic Blood Pressure", "BMI", "Insulin"],
        "Optimal Level": ["70 – 99 mg/dL", "< 80 mm Hg", "18.5 – 24.9 kg/m²", "16 – 166 mu U/ml"],
        "Borderline / At-Risk": ["100 – 125 mg/dL", "80 – 89 mm Hg", "25.0 – 29.9 kg/m²", "Elevated relative to glucose"],
        "Diagnostic Alert": ["≥ 126 mg/dL", "≥ 90 mm Hg", "≥ 30.0 kg/m²", "Significant insulin resistance"]
    }
    st.table(pd.DataFrame(ref_table))
    st.info("Disclaimer: This tool is intended for screening and educational purposes and does not replace official physician diagnosis.")
