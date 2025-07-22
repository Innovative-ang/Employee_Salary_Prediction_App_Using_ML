import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from streamlit_lottie import st_lottie
import requests
import base64

# ------Page Setup and Theming Block -----
st.set_page_config(page_title="Employee Salary Prediction Using Machine Learning", layout="centered")

# Background image handling
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_filename = "dark-workspace.jpg"  # Set your local background image file
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{get_base64_of_bin_file(img_filename)}");
    background-size: cover;
    background-repeat: no-repeat;
    background-position: center center;
    background-attachment: fixed;
}}
/* ====== Theme Customization ====== */
.custom-main-title {{
    background: #1565c0;
    color: #fff !important;
    font-size: 2.7rem;
    font-weight: bold;
    border-radius: 16px;
    text-align: center;
    padding: 17px 10px;
    margin-bottom: 25px;
    box-shadow: 0 4px 18px #16407036;
}}
[data-testid="stForm"] {{
    background: #23272aee !important;
    border-radius: 15px !important;
    box-shadow: 0 2px 16px #23272a37;
    padding: 18px 5px 8px 5px;
}}
label, .stSelectbox label, .stNumberInput label, .stTextInput label {{
    color: #fff !important;
    font-weight: 600;
}}
.stTextInput>div>input, .stNumberInput>div>input, .stSelectbox>div>div {{
    color: #fff !important;
}}
.stTable th, .stTable td, .stDataFrame th, .stDataFrame td {{
    color: #181818 !important;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ------------------------------------#
# Lottie Animation and Title Section  #
# ------------------------------------#
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        return None

with st.container():
    col_anim, col_txt = st.columns([2, 3])
    with col_anim:
        office_lottie = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_9cyyl8i4.json")
        if office_lottie:
            st_lottie(office_lottie, speed=1, width=230, height=170, key="office_anim", loop=True)
    with col_txt:
        st.markdown(
            '<div class="custom-main-title">Employee Salary Prediction</div>',
            unsafe_allow_html=True
        )
        st.write(
            "Accurately predict if an employee's annual income is **'>50K'** or **'<=50K'** using demographic and employment features."
        )

# -----------------------------------#
# Data Loading and Model Preparation #
# -----------------------------------#
@st.cache_data
def load_and_prepare_data():
    data = pd.read_csv(r"C:\Users\anura\Downloads\IBMSKILLBUILD Project\adult 3.csv")
    categorical = ['workclass', 'education', 'marital-status', 'occupation',
                   'relationship', 'race', 'gender', 'native-country']
    X = data.drop('income', axis=1)
    y = data['income']
    encoders = {}
    for col in categorical:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    # Train with best model: Gradient Boosting
    model = GradientBoostingClassifier()
    model.fit(X, y)
    return X, y, categorical, encoders, model

X, y, categorical, encoders, model = load_and_prepare_data()

# --------------------------------------#
# Tabbed UI: Single & Batch Prediction  #
# --------------------------------------#
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# ----- Single Prediction Form Block -----
with tab1:
    with st.form("input_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", 17, 90, 28)
            workclass = st.selectbox("Workclass", options=encoders['workclass'].classes_)
            education = st.selectbox("Education", options=encoders['education'].classes_)
            educational_num = st.number_input("Education Number", 1, 16, 12)
            marital_status = st.selectbox("Marital Status", options=encoders['marital-status'].classes_)
            occupation = st.selectbox("Occupation", options=encoders['occupation'].classes_)
            relationship = st.selectbox("Relationship", options=encoders['relationship'].classes_)
        with c2:
            race = st.selectbox("Race", options=encoders['race'].classes_)
            gender = st.selectbox("Gender", options=encoders['gender'].classes_)
            fnlwgt = st.number_input("fnlwgt", 10000, 700000, 336951)
            capital_gain = st.number_input("Capital Gain", 0, 99999, 0)
            capital_loss = st.number_input("Capital Loss", 0, 99999, 0)
            hours_per_week = st.number_input("Hours per Week", 1, 99, 40)
            native_country = st.selectbox("Native Country", options=encoders['native-country'].classes_)

        predict_btn = st.form_submit_button("Predict Salary Class")
        
        if predict_btn:
            input_dict = {
                'age': age,
                'workclass': workclass,
                'fnlwgt': fnlwgt,
                'education': education,
                'educational-num': educational_num,
                'marital-status': marital_status,
                'occupation': occupation,
                'relationship': relationship,
                'race': race,
                'gender': gender,
                'capital-gain': capital_gain,
                'capital-loss': capital_loss,
                'hours-per-week': hours_per_week,
                'native-country': native_country
            }
            input_df = pd.DataFrame([input_dict])
            for col in categorical:
                input_df[col] = encoders[col].transform(input_df[col].astype(str))
            input_df = input_df[X.columns]
            pred = model.predict(input_df)[0]
            result = ">50K" if str(pred).strip() in [">50K", "&gt;50K"] else "<=50K"
            color = "#37b24d" if result == ">50K" else "#1971c2"
            st.markdown(f"<h2 style='color:{color};'>Predicted: {result}</h2>", unsafe_allow_html=True)
            st.table(pd.DataFrame([input_dict]))

# ---------- Batch Prediction Block ----------
with tab2:
    st.markdown("#### Batch CSV Prediction")
    st.info("Upload a CSV (same columns as your training data, except **no 'income' column** required).")
    batch_file = st.file_uploader("Choose CSV file for batch prediction", type=['csv'])
    if batch_file:
        try:
            df_input = pd.read_csv(batch_file)
            for col in categorical:
                df_input[col] = encoders[col].transform(df_input[col].astype(str))
            missing_cols = [col for col in X.columns if col not in df_input.columns]
            if missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}")
            else:
                df_input = df_input[X.columns]
                preds = model.predict(df_input)
                df_result = df_input.copy()
                df_result["Predicted Income"] = preds
                df_result["Predicted Income"] = df_result["Predicted Income"].replace(
                    {"&gt;50K": ">50K", "&lt;=50K": "<=50K"}
                )
                st.success("Batch prediction complete! Download your results below.")
                st.dataframe(df_result.head(10))
                csv = df_result.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions CSV", data=csv, file_name="salary_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error: {e}")

# -------------- Info Expander Block --------------
with st.expander("About this app"):
    st.write(
        "This app is designed for professional HR, hiring, and analytics workflows. "
        "UI is fully responsive, with accurate batch and single-candidate predictions."
    )
