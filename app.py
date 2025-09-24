import streamlit as st
import joblib
import pandas as pd
import os
import gdown

@st.cache_resource
def load_rf_model():
    if not os.path.exists("random_forest_high_spend.joblib"):
        gdown.download("https://drive.google.com/uc?export=download&id=1-Ba7YlUb_32oA_wlC32BRJ0RbwYotjFc",
                       "random_forest_high_spend.joblib", quiet=False)
    return joblib.load("random_forest_high_spend.joblib")

@st.cache_resource
def load_xgb_model():
    if not os.path.exists("xgboost_churn.joblib"):
        gdown.download("https://drive.google.com/uc?export=download&id=1GJgCpzISeKoUrT5GuUUN9jcDxkoPzrT9",
                       "xgboost_churn.joblib", quiet=False)
    return joblib.load("xgboost_churn.joblib")

cat_encoders = {
    "Customer_Category": {'Student': 0, 'Teenager': 1, 'Middle-Aged': 2, 'Senior Citizen': 3, 'Retiree': 4},
    "City": {'New York': 0, 'Chicago': 1, 'Los Angeles': 2, 'San Francisco': 3,
             'Boston': 4, 'Dallas': 5, 'Seattle': 6, 'Houston': 7, 'Miami': 8},
    "Store_Type": {'Department Store': 0, 'Warehouse Club': 1, 'Pharmacy': 2, 'Supermarket': 3},
    "Payment_Method": {'Credit Card': 0, 'Debit Card': 1, 'Cash': 2}
}

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
    st.markdown("<div class='model-name-rf'>🌲 Random Forest (Accuracy: 96%) — High Spend</div>", unsafe_allow_html=True)
    basket_size = st.number_input("Basket Size (No. of Items Purchased)", min_value=1, step=1, key="rf_basket")
    avg_item_price = st.number_input("Average Item Price", min_value=1, step=1, key="rf_avg")
    dow_label = st.selectbox("Day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], key="rf_dow")
    dow_map = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}
    dayofweek = dow_map[dow_label]
    hour = st.number_input("Hour of Transaction (0-23)", min_value=0, max_value=23, step=1, key="rf_hour")
    rf_input = pd.DataFrame([{
        "Basket_Size": int(basket_size),
        "Avg_Item_Price": float(avg_item_price),
        "DayOfWeek": int(dayofweek),
        "Hour": int(hour),
        "Basket_Value": float(basket_size) * float(avg_item_price)
    }])
    if st.button("Predict High Spend Flag", key="rf_btn"):
        rf_model = load_rf_model()
        rf_input = rf_input.reindex(columns=rf_model.feature_names_in_, fill_value=0)
        rf_pred = rf_model.predict(rf_input)[0]
        if rf_pred == 1:
            st.markdown(f"<div class='result result-good'>💸 High Spender</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result result-bad'>🛒 Normal Spender</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='model-name-xgb'>🔥 XGBoost (Accuracy: 83%) — Churn Prediction</div>", unsafe_allow_html=True)
    customer_category = st.selectbox("Customer Category", list(cat_encoders["Customer_Category"].keys()), key="xgb_cat")
    city = st.selectbox("City", list(cat_encoders["City"].keys()), key="xgb_city")
    store_type = st.selectbox("Store Type", list(cat_encoders["Store_Type"].keys()), key="xgb_store")
    payment_method = st.selectbox("Payment Method", list(cat_encoders["Payment_Method"].keys()), key="xgb_pay")
    recency = st.number_input("Recency (Days since Last Purchase)", min_value=0, step=1, key="xgb_recency")
    frequency = st.number_input("Frequency (No. of Purchases)", min_value=0, step=1, key="xgb_freq")
    monetary = st.number_input("Monetary (Total Spend)", min_value=0.0, step=1.0, key="xgb_monetary")
    if st.button("Predict Churn", key="xgb_btn"):
        xgb_model = load_xgb_model()
        xgb_input = pd.DataFrame([{
            "Recency": float(recency),
            "Frequency": int(frequency),
            "Monetary": float(monetary),
            "Customer_Category": cat_encoders["Customer_Category"][customer_category],
            "City": cat_encoders["City"][city],
            "Store_Type": cat_encoders["Store_Type"][store_type],
            "Payment_Method": cat_encoders["Payment_Method"][payment_method]
        }])
        xgb_input = xgb_input.reindex(columns=xgb_model.feature_names_in_, fill_value=0)
        churn_pred = xgb_model.predict(xgb_input)[0]
        if churn_pred == 1:
            st.markdown(f"<div class='result result-bad'>❌ Likely to Churn</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result result-good'>✅ Likely to Stay</div>", unsafe_allow_html=True)
