# RetailMaster
# 🛒 Retail Predictor App

This **Retail ML Prediction App** helps businesses analyze customer behavior using **machine learning models**.  
It provides **two predictive services** with an interactive Streamlit interface:

---

## 🚀 Features

### 1. High Spending Prediction — **Random Forest (96% Accuracy)**
Predicts whether a customer belongs to the top 5% of high spenders.  
**Input Features:**
- Basket Size (number of items purchased)  
- Average Item Price  
- Day of Week (Mon–Sun)  
- Hour of Transaction (0–23)  

---

### 2. Customer Churn Prediction — **XGBoost (83% Accuracy)**
Estimates the likelihood of a customer leaving (churn) or staying.  
**Input Features:**
- Recency (days since last purchase)  
- Frequency (number of purchases)  
- Monetary Value (total spend)  
- Customer Category (e.g., Student, Retiree, Professional)  
- City  
- Store Type  
- Payment Method  

---

## 🖼 App Design
- **Dark theme** with glowing model names.  
- **Tab-based UI**: One tab for Random Forest and one for XGBoost.  
- Results displayed with clear green/red highlights for quick decision-making.  

---

## 📂 Project Structure
Retail-Predictor/
│
├── app.py # Main Streamlit app
├── random_forest_high_spend.pkl # Trained Random Forest model
├── xgboost_churn.pkl # Trained XGBoost model
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## ⚙️ Installation & Usage

### 1. Clone the repo
git clone https://github.com/saadbutt00/RetailMaster.git
cd RetailMaster

### 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run the app
streamlit run app.py

---

### 📊 Dataset
- A synthetic dataset was used to train the models (Random Forest and XGBoost).
- This ensures no privacy concerns and demonstrates predictive modeling.

### ✅ Tech Stack
**Streamlit** — Interactive user interface
**Pandas** — Data processing
**Scikit-learn** — Random Forest model
**XGBoost** — Churn prediction model

### 📌 Notes
- Random Forest is tuned to detect high-spending customers.
- XGBoost is tuned to detect customer churn risk.

Both models were pre-trained and saved as .pkl files for deployment.
