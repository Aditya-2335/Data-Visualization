import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# -----------------------------
# Load Model & Dataset
# -----------------------------
@st.cache_data(ttl=3600)
def load_data():
    return pd.read_csv("health_data.csv")

@st.cache_resource
def load_model():
    try:
        return joblib.load("cardio_model.pkl")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

df = load_data()
model = load_model()

# -----------------------------
# Page Config & Custom CSS
# -----------------------------
st.set_page_config(
    page_title="💓 Cardiovascular Health Dashboard",
    page_icon="💓",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #F8F9FA;
    }
    h1, h2, h3, h4 {
        color: #D72638;
    }
    .stButton>button {
        background-color: #D72638;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #b81d2e;
        color: white;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Custom Tab Styling */
    div[data-baseweb="tab-list"] {
        gap: 20px;
        justify-content: center;
        margin-top: 15px;
        border-bottom: 2px solid #ddd;
        padding-bottom: 5px;
    }
    div[data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #555 !important;
        padding: 8px 20px;
        border-radius: 8px 8px 0 0;
        background-color: #f2f2f2;
        transition: all 0.3s ease;
    }
    div[data-baseweb="tab"][aria-selected="true"] {
        color: white !important;
        background-color: #D72638 !important;
        border-bottom: 3px solid #D72638 !important;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.15);
    }
    div[data-baseweb="tab"]:hover {
        background-color: #b81d2e !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
st.sidebar.title("💓 Cardio Dashboard")
st.sidebar.markdown("Navigate the sections below:")
st.sidebar.markdown("**Tabs:** Overview • Prediction • Visualization")

# Tabs
tab1, tab2, tab3 = st.tabs(["📘 Overview", "🩺 Prediction", "📊 Data Visualization & Analysis"])

# -----------------------------
# TAB 1: Overview
# -----------------------------
with tab1:
    st.title("📘 Overview")
    st.write("""
    Welcome to the **Cardiovascular Health Prediction Dashboard**.  
    This app allows you to **analyze**, **visualize**, and **predict** cardiovascular disease risk 
    using machine learning.
    """)

    st.subheader("📈 Quick Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Average Age (days)", int(df["age"].mean()))
    col3.metric("Cardio Cases", int(df["cardio"].sum()))

    st.markdown("### 🧾 Data Sample")
    # Display readable gender and cardio labels
    df_display = df.copy()
    df_display["gender"] = df_display["gender"].map({1: "Male", 0: "Female"})
    df_display["cardio"] = df_display["cardio"].map({1: "Has Disease", 0: "No Disease"})
    st.dataframe(df_display.head())

    st.markdown("### 🧮 Cardio Distribution")
    cardio_counts = df["cardio"].value_counts().reset_index()
    cardio_counts.columns = ["Cardio (0=No,1=Yes)", "Count"]
    fig = px.pie(cardio_counts, names="Cardio (0=No,1=Yes)", values="Count",
                 color_discrete_sequence=["#28A745", "#D72638"],
                 title="Cardiovascular Disease Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Model Information**
    - Algorithm: Random Forest Classifier  
    - Target: `cardio` (1 = disease, 0 = healthy)  
    - Input Features: Age, Blood Pressure, Cholesterol, Glucose, Lifestyle Habits
    """)

# -----------------------------
# TAB 2: Prediction (Improved)
# -----------------------------
with tab2:
    st.title("🩺 Predict Cardiovascular Disease Risk")

    st.write("Enter patient details below to predict cardiovascular disease risk:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (days)", min_value=1000, max_value=30000, value=18000)
        gender = st.selectbox("Gender", ["Female", "Male"])
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=165)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        chol = st.selectbox("Cholesterol", ["Normal", "Above Normal", "Well Above Normal"])

    with col2:
        sbp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
        dbp = st.number_input("Diastolic BP", min_value=50, max_value=130, value=80)
        gluc = st.selectbox("Glucose", ["Normal", "Above Normal", "Well Above Normal"])
        smoke = st.selectbox("Smoking", ["No", "Yes"])
        alcohol = st.selectbox("Alcohol Intake", ["No", "Yes"])
        active = st.selectbox("Physical Activity", ["No", "Yes"])

    # Derived metrics
    bmi = weight / ((height / 100) ** 2)
    age_years = age / 365
    st.markdown(f"**🧮 Calculated BMI:** {bmi:.1f}")
    st.markdown(f"**📅 Age in years:** {age_years:.1f}")

    # Convert categorical input
    gender = 1 if gender == "Male" else 0
    chol = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[chol]
    gluc = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}[gluc]
    smoke = 1 if smoke == "Yes" else 0
    alcohol = 1 if alcohol == "Yes" else 0
    active = 1 if active == "Yes" else 0

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "Systolic blood pressure": sbp,
        "Diastolic blood pressure": dbp,
        "cholesterol": chol,
        "glucose": gluc,
        "smoking": smoke,
        "alcohol intake": alcohol,
        "physical activity": active
    }])

    st.markdown("---")

    if st.button("🔍 Predict Risk"):
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Predicted Cardio Risk (%)"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "#D72638"},
                       'steps': [
                           {'range': [0, 50], 'color': "#28A745"},
                           {'range': [50, 100], 'color': "#D72638"}
                       ]}
            ))
            st.plotly_chart(fig, use_container_width=True)

            if prediction == 1:
                st.error(f"⚠️ **High Risk Detected!** Probability: {probability:.2f}")
                st.markdown("<h4 style='color:#D72638;'>Consult your doctor for preventive advice 🩺</h4>", unsafe_allow_html=True)
            else:
                st.success(f"✅ **Low Risk.** Probability: {probability:.2f}")
                st.markdown("<h4 style='color:#28A745;'>Maintain your healthy habits 🌿</h4>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

    if hasattr(model, "feature_importances_"):
        st.markdown("### 🧠 Feature Importance")
        importance = pd.DataFrame({
            "Feature": model.feature_names_in_,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        fig = px.bar(importance, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    
# -----------------------------
# TAB 3: Data Visualization & Analysis (Improved with readable filters)
# -----------------------------
with tab3:
    st.title("📊 Data Visualization & Correlation Analysis")

    # Add readable gender and cardio labels
    df["gender_label"] = df["gender"].map({1: "Male", 0: "Female"})
    df["cardio_label"] = df["cardio"].map({1: "Has Disease", 0: "No Disease"})

    st.sidebar.subheader("🎛 Data Filters")

    gender_options = ["Male", "Female"]
    selected_gender = st.sidebar.multiselect("Filter by Gender", gender_options, default=gender_options)

    cardio_options = ["No Disease", "Has Disease"]
    selected_cardio = st.sidebar.multiselect("Filter by Cardio Status", cardio_options, default=cardio_options)

    filtered_df = df[
        (df["gender_label"].isin(selected_gender)) &
        (df["cardio_label"].isin(selected_cardio))
    ]

    st.markdown("### 🧮 Choose Columns and Chart Type")
    all_columns = df.columns.tolist()
    x_axis = st.selectbox("X-axis", all_columns, index=1)
    y_axis = st.selectbox("Y-axis (optional)", ["None"] + all_columns, index=0)
    plot_type = st.selectbox("Plot Type", ["Histogram", "Scatter", "Boxplot", "Correlation Heatmap"])

    if plot_type == "Histogram":
        fig = px.histogram(filtered_df, x=x_axis, nbins=30, color="cardio_label",
                           color_discrete_sequence=["#28A745", "#D72638"],
                           title=f"Distribution of {x_axis}")
    elif plot_type == "Scatter" and y_axis != "None":
        fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color="cardio_label",
                         color_discrete_sequence=["#28A745", "#D72638"],
                         title=f"{x_axis} vs {y_axis}")
    elif plot_type == "Boxplot" and y_axis != "None":
        fig = px.box(filtered_df, x=x_axis, y=y_axis, color="cardio_label",
                     color_discrete_sequence=["#28A745", "#D72638"],
                     title=f"{y_axis} by {x_axis}")
    elif plot_type == "Correlation Heatmap":
        corr = filtered_df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                        title="Correlation Heatmap")
    else:
        fig = px.histogram(filtered_df, x=x_axis, nbins=30, title=f"Distribution of {x_axis}")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📈 Summary Statistics")
    st.dataframe(filtered_df.describe().T)

    st.markdown("### 🔗 Top Correlations with `cardio`")
    corr = df.corr(numeric_only=True)["cardio"].abs().sort_values(ascending=False)
    st.dataframe(corr.head(5).to_frame("Correlation Strength"))

    st.markdown("### 💬 Observations")
    st.write("""
    - **Age**, **Blood Pressure**, and **Cholesterol** often show strong correlation with cardiovascular disease.  
    - Lifestyle indicators like **smoking**, **alcohol**, and **activity** provide secondary insights.  
    - Use filters on the left sidebar to explore patterns by gender or condition.  
    """)

# -----------------------------
# Footer / Branding
# -----------------------------
st.markdown("---")
st.caption("© 2025 CardioPredict") 