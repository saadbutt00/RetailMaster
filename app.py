import streamlit as st
import pickle
import pandas as pd
import os
import gdown
import sys

def download_model(url, filename):
    if not os.path.exists(filename):
        gdown.download(url, filename, quiet=False)

# ✅ Download models from Google Drive
download_model("https://drive.google.com/uc?export=download&id=1ickiXA8ZakRHI0UPxEFToFVq5FRwr4nv",
               "random_forest_high_spend.pkl")
download_model("https://drive.google.com/uc?export=download&id=1m-VuAaseZ5Gn2yn5hJgMcknw17EC9Sxy",
               "xgboost_churn.pkl")

def add_basket_value(X):
    X = X.copy()
    X["Basket_Value"] = X["Basket_Size"] * X["Avg_Item_Price"]
    return X

def clip_features(X):
    X = X.copy()
    X["Recency"] = X["Recency"].clip(50, 700)
    X["Frequency"] = X["Frequency"].clip(0, 20)
    X["Monetary"] = X["Monetary"].clip(50, 550)
    return X

# ✅ Register custom functions so pickle can find them
sys.modules['__main__'].clip_features = clip_features
sys.modules['__main__'].add_basket_value = add_basket_value

with open("random_forest_high_spend.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("xgboost_churn.pkl", "rb") as f:
    xgb_model = pickle.load(f)

st.set_page_config(page_title="Retail Predictor", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        background: radial-gradient(circle at 10% 10%, rgba(10,10,10,0.9), rgba(0,0,0,1)), #000000;
        color: #e6f4f1;
        font-family: 'Montserrat', sans-serif;
    }
    .model-name-rf {
        font-size:20px;font-weight:700;color:#00ff9a;
        text-shadow:0 0 12px #00ff9a, 0 0 18px #00ff9a;
        margin-bottom:15px;
    }
    .model-name-xgb {
        font-size:20px;font-weight:700;color:#ff2e44;
        text-shadow:0 0 12px #ff2e44, 0 0 18px #ff2e44;
        margin-bottom:15px;
    }
    .result {margin-top:14px;padding:12px;border-radius:10px;font-weight:700;}
    .result-good {background:rgba(0,255,150,0.08);color:#00ff9a;text-shadow:0 0 8px #00ff9a;}
    .result-bad {background:rgba(255,40,70,0.08);color:#ff2e44;text-shadow:0 0 8px #ff2e44;}
    </style>
    """,
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["🌲 Random Forest", "🔥 XGBoost"])

with tab1:
    st.write('''This Retail Predictor App helps businesses analyze customer behavior using machine learning models. 
    It provides two main predictive services:
- High Spending Prediction (Random Forest, 96% Accuracy):
- Predicts whether a customer belongs to the top 5% of high spenders based on features like:

**Features:**

    - Basket Size (number of items purchased)
    - Average Item Price
    - Day of Week of purchase
    - Hour of Transaction''')

    st.markdown("<div class='model-name-rf'>🌲 Random Forest (Accuracy: 96%) — High Spend</div>", unsafe_allow_html=True)

    basket_size = st.number_input("Basket Size (No. of Items Purchased)", min_value=1, step=1, key="rf_basket")
    avg_item_price = st.number_input("Average Item Price", min_value=1, step=1, key="rf_avg")
    dow_label = st.selectbox("Day", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], key="rf_dow")
    dow_map = {"Mon":1,"Tue":2,"Wed":3,"Thu":4,"Fri":5,"Sat":6,"Sun":7}
    dayofweek = dow_map[dow_label]
    hour = st.number_input("Hour of Transaction (0-23)", min_value=0, max_value=23, step=1, key="rf_hour")

    rf_input = pd.DataFrame([{
        "Basket_Size": int(basket_size),
        "Avg_Item_Price": float(avg_item_price),
        "DayOfWeek": int(dayofweek),
        "Hour": int(hour)
    }])

    if st.button("Predict High Spend Flag", key="rf_btn"):
        rf_pred = rf_model.predict(rf_input)[0]
        if rf_pred == 1:
            st.markdown(f"<div class='result result-good'>💸 High Spender</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result result-bad'>🛒 Normal Spender</div>", unsafe_allow_html=True)

with tab2:
    st.write('''Customer Churn Prediction (XGBoost, 83% Accuracy):
- Estimates the likelihood of a customer leaving (churn) or staying based on features such as:

    - Recency (days since last purchase)
    - Frequency (number of purchases)
    - Monetary Value (total spend)
    - Customer Category (e.g., Student, Professional)
    - City
    - Store Type
    - Payment Method
    
With an interactive interface, the app lets users input customer data and instantly get predictive insights
to support marketing, loyalty programs, and retention strategies.''')

    st.markdown("<div class='model-name-xgb'>🔥 XGBoost (Accuracy: 83%) — Churn Prediction</div>", unsafe_allow_html=True)

    customer_category = st.selectbox("Customer Category", ['Student', 'Teenager', 'Middle-Aged', 'Senior Citizen', 'Retiree'], key="xgb_cat")
    city = st.selectbox("City", ['New York', 'Chicago', 'Los Angeles', 'San Francisco', 'Boston', 'Dallas', 'Seattle', 'Houston', 'Miami'], key="xgb_city")
    store_type = st.selectbox("Store Type", ['Department Store','Warehouse Club','Pharmacy','Supermarket'], key="xgb_store")
    payment_method = st.selectbox("Payment Method", ['Credit Card','Debit Card','Cash'], key="xgb_pay")

    recency = st.number_input("Recency (Days since Last Purchase)", min_value=0, step=1, key="xgb_recency")
    frequency = st.number_input("Frequency (No. of Purchases)", min_value=0, step=1, key="xgb_freq")
    monetary = st.number_input("Monetary (Total Spend)", min_value=0.0, step=1.0, key="xgb_monetary")

    xgb_input = pd.DataFrame([{
        "Recency": float(recency),
        "Frequency": int(frequency),
        "Monetary": float(monetary),
        "Customer_Category": customer_category,
        "City": city,
        "Store_Type": store_type,
        "Payment_Method": payment_method
    }])

    if st.button("Predict Churn", key="xgb_btn"):
        churn_pred = xgb_model.predict(xgb_input)[0]
        if churn_pred == 1:
            st.markdown(f"<div class='result result-bad'>❌ Likely to Churn</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result result-good'>✅ Likely to Stay</div>", unsafe_allow_html=True)

