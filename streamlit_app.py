import streamlit as st
import pandas as pd
import joblib
import json
import math
import matplotlib.pyplot as plt
import charts
st.set_page_config(page_title="AI Overdependence Predictor (Gauge A)", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; font-size: 48px; font-weight: 700;'>
        🤖 AI Dependency Prediction Dashboard
    </h1>
    <p style='text-align: center; font-size: 16px; color: #b3b3b3; margin-top: -10px;'>
        The AI Dependency Predictor evaluates your level of reliance on AI—categorizing you as highly dependent, moderately dependent, or minimally dependent—based on your responses to a series of questions.
    </p>
    """,
    unsafe_allow_html=True
)


MODEL_PATH = "models/xgb_over_model_noleak_v2.pkl"
SCALER_PATH = "models/scaler_noleak_v2.pkl"
FEATURES_CSV = "data/features_header_noleak_v2.csv"
MODEL = joblib.load(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)
FEATURES = list(pd.read_csv(FEATURES_CSV).columns)

TOP10 = ["Problem-Solving [Prob3: I often skip brainstorming or planning because I expect AI to do most of the work.]", "Problem-Solving [Prob2: I have become less persistent in solving academic challenges because I know AI will help.]", "Alienation [AL6: AI capabilities make my own academic growth feel less valuable.]", "Creative Thinking [Creat1: I use AI tools even when I could think of original ideas yourself.]", "Creative Thinking [Creat3: My assignments are less original because I adapt AI-generated ideas instead of developing my own.]", "Self-Confidence [SC1: I doubt my ability to complete tasks without AI support.]", "Alienation [AL2: I worry that AI-generated assignments donot truly reflect what I've learned.]", "Collaboration [Collab1: I prefer working alone with AI tools rather than discussing ideas with peers.]", "Communication [Comm2: I feel confident when explaining complex topics to others in class discussions or presentations.]", "Alienation [AL8: Using AI for group work makes collaboration feel less authentic.]"]

st.sidebar.title("👤 Profile (save before predicting)")
st.sidebar.markdown("📝 Please fill your demographic info and **Save Profile** before answering the 10 short questions.")

gender = st.sidebar.selectbox("Gender", ["Female", "MISSING", "Male"], index=0)
age = st.sidebar.selectbox("Age group", ["18-24", "25-34", "35-44", "MISSING"], index=0)
education = st.sidebar.selectbox("Educational qualification", ["Bachelor Degree", "High School Diploma", "MISSING", "Master Degree", "PhD"], index=0)
frequency = st.sidebar.selectbox("How often do you use Generative AI tools?", ["Never","Rarely","Monthly","Weekly","Daily","Prefer not to say"], index=2)

if "profile_saved" not in st.session_state:
    st.session_state["profile_saved"] = False
    st.session_state["profile"] = {}

if st.sidebar.button("Save profile"):
    profile = dict()
    for f in FEATURES:
        profile[f] = 3.0 if not (f.startswith("Gender_") or f.startswith("Age_") or f.startswith("Educational_Qualification_") or f.startswith("Frequency_of_Use_of_GenAI_Tools")) else 0.0
    def internal_suffix(val):
        if val == "Prefer not to say":
            return "MISSING"
        return val
    g_key = "Gender_" + internal_suffix(gender)
    a_key = "Age_" + internal_suffix(age)
    e_key = "Educational_Qualification_" + internal_suffix(education)
    for f in FEATURES:
        if f == g_key:
            profile[f] = 1.0
        if f == a_key:
            profile[f] = 1.0
        if f == e_key:
            profile[f] = 1.0
        if f.startswith("Frequency_of_Use_of_GenAI_Tools") and frequency in f:
            profile[f] = 1.0
        if f.startswith("Frequency_of_Use_of_GenAI_Tools") and frequency == "Prefer not to say" and f.endswith("MISSING"):
            profile[f] = 1.0
    st.session_state["profile_saved"] = True
    st.session_state["profile"] = profile
    st.sidebar.success("Profile saved. Now answer the 10 questions and press Predict.")

tabs = st.tabs(["📈Data Visualizations", "⚡AI Dependency predictor"])

with tabs[0]:
    charts.show()

with tabs[1]:
    st.header("⚡AI dependence Score prediction")
    st.write("Answer the 10 short questions below (1 = Strongly disagree, 5 = Strongly agree).")

    if not st.session_state.get("profile_saved", False):
        st.warning("Please fill the profile in the sidebar and click **Save profile** before answering the questions.")

    responses = dict()

    for i, q in enumerate(TOP10):
        st.markdown("**Q" + str(i+1) + ". " + q + "**")
        responses[q] = st.number_input("", min_value=1.0, max_value=5.0, value=3.0, step=0.5, key="ans_" + str(i))

    st.markdown("---")
    cols = st.columns([1,1])
    with cols[0]:
        if st.button("Predict"):
            if not st.session_state.get("profile_saved", False):
                st.error("You must save your profile in the sidebar before predicting.")
            else:
                inp = dict(st.session_state["profile"])
                for k,v in responses.items():
                    if k in inp:
                        inp[k] = v
                    else:
                        matches = [f for f in FEATURES if f.startswith(k.split('[')[0].strip())]
                        if matches:
                            inp[matches[0]] = v
                for f in FEATURES:
                    if f not in inp:
                        inp[f] = 3.0
                X = pd.DataFrame([inp], columns=FEATURES)
                Xs = SCALER.transform(X)
                pred = MODEL.predict(Xs)[0]
                st.metric("Predicted overdependence (0-100%)", f"{pred:.1f}%")                
                if pred <= 33.33:
                    level = "Low"
                elif pred <= 66.66:
                    level = "Medium"
                else:
                    level = "High"
                st.markdown(f"**Interpretation:** {level} overdependence on AI")
    with cols[1]:
        st.info("Flow: Fill sidebar → Save profile → Answer questions → Predict")
        st.write("Model performance (test set):")
        report = json.load(open("data/training_report_noleak_v2.json"))
        st.write("R² = " + str(report.get("over_r2_v2")) + " (test)")
