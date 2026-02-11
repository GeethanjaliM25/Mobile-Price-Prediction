import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# ===============================
# Page Config (VERY IMPORTANT for UI)
# ===============================
st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="wide"
)

# ===============================
# Custom CSS (UI Polish)
# ===============================
st.markdown("""
<style>
    .main {
        background-color: #f9fafb;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    .stButton button {
        background-color: #2563eb;
        color: white;
        font-size: 18px;
        padding: 0.6em 2em;
        border-radius: 10px;
    }
    .stButton button:hover {
        background-color: #1e40af;
    }
    .prediction-box {
        padding: 20px;
        background-color: #ecfeff;
        border-radius: 12px;
        border-left: 8px solid #06b6d4;
        font-size: 22px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# App Header
# ===============================
st.title("📱 Mobile Price Prediction App")
st.markdown(
    "Predict **Mobile Price Category** using Machine Learning models based on phone specifications."
)

st.divider()

# ===============================
# Load & Train Model
# ===============================
@st.cache_data
def train_model(model_name):
    df = pd.read_csv("Mobile_data.csv")

    # Data cleaning
    df["px_height"] = np.where(df["px_height"] == 0, 125, df["px_height"])
    df["sc_w"] = np.where(df["sc_w"] == 0, 2, df["sc_w"])

    X = df.drop(columns="Price")
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=88
    )

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)

    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(
            max_depth=9, min_samples_split=200, random_state=88
        )

    else:
        model = RandomForestClassifier(
            n_estimators=250, max_depth=9, min_samples_split=200, random_state=88
        )

    model.fit(X_train, y_train)
    return model, X.columns

# ===============================
# Sidebar – Model Selection
# ===============================
st.sidebar.header("⚙️ Model Settings")

model_choice = st.sidebar.selectbox(
    "Choose ML Model",
    ("Logistic Regression", "Decision Tree", "Random Forest")
)

st.sidebar.info(
    "💡 **Tip:** Random Forest usually gives better accuracy for this dataset."
)

model, feature_names = train_model(model_choice)

# ===============================
# User Input Section
# ===============================
st.subheader("🔢 Enter Mobile Specifications")

col1, col2, col3 = st.columns(3)

with col1:
    battery_power = st.number_input("🔋 Battery Power (mAh)", 500, 6000)
    clock_speed = st.number_input("⚡ Clock Speed (GHz)", 0.5, 5.0)
    n_cores = st.number_input("🧠 CPU Cores", 1, 12)
    ram = st.number_input("💾 RAM (MB)", 256, 12000)

with col2:
    fc = st.number_input("🤳 Front Camera (MP)", 0, 50)
    pc = st.number_input("📷 Primary Camera (MP)", 0, 108)
    int_memory = st.number_input("📂 Internal Memory (GB)", 2, 512)
    mobile_wt = st.number_input("⚖️ Mobile Weight (grams)", 80, 300)

with col3:
    px_height = st.number_input("🖥️ Pixel Height", 0, 3000)
    px_width = st.number_input("🖥️ Pixel Width", 0, 3000)
    sc_h = st.number_input("📐 Screen Height (cm)", 5, 20)
    sc_w = st.number_input("📐 Screen Width (cm)", 2, 15)
    talk_time = st.number_input("📞 Talk Time (hours)", 1, 30)

# ===============================
# Prediction
# ===============================
st.divider()

user_data = pd.DataFrame([[ 
    battery_power, clock_speed, fc, int_memory, 0.5,
    mobile_wt, n_cores, pc, px_height, px_width,
    ram, sc_h, sc_w, talk_time
]], columns=feature_names)

if st.button("🔮 Predict Mobile Price"):
    prediction = model.predict(user_data)[0]

    price_map = {
        0: "Low Cost 📉",
        1: "Medium Cost 💰",
        2: "High Cost 💎",
        3: "Very High Cost 👑"
    }

    st.markdown(
        f"""
        <div class="prediction-box">
            Predicted Price Category: {price_map[prediction]}
        </div>
        """,
        unsafe_allow_html=True
    )

# ===============================
# Footer
# ===============================
st.divider()
st.caption("👩‍💻 Built using Streamlit & Machine Learning | BE Mini / Major Project")
