# Telco Customer Churn Predictor

Streamlit dashboard + FastAPI backend for churn prediction using LightGBM.

- Dashboard → port 7860 (visible in browser)
- API → http://localhost:8000
- Hugging face live - [https://huggingface.co/spaces/vinaykumarkv/telco-churn-predictor](https://huggingface.co/spaces/vinaykumarkv/telco-churn-predictor) 


**Features**
- Real-time churn probability using tuned LightGBM
- Interactive form with all Telco dataset features
- Risk categorization (Low / Medium / High)
- Backend: FastAPI • Frontend: Streamlit • Model saved via joblib

**Key Insights from SHAP**
- 2-year contracts = strongest churn prevention
- Fiber optic + electronic check + high monthly charges = major risk drivers

Built as a portfolio project — code & notebook in the repo files.
