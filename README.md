
# 🏦 Loan Acceptance Predictor

A machine learning project to predict if a loan application is likely to be **accepted** based on applicant information.

## 🚀 Quick Overview

- **Tech Stack**: Python, Scikit-learn, Streamlit
- **Goal**: Predict loan approval using historical applicant data
- **Target Variable**: `Loan_Status` (1 = Accepted, 0 = Rejected)

## 📁 Main Files

- `notebooks/`: EDA, preprocessing, and model training notebooks
- `model/`: Serialized trained model and scaler
- `app.py`: Streamlit web app
- `requirements.txt`: Required Python packages

## 🛠️ Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🖥️ Web App Features

- Input form for user details (age, income, education, etc.)
- Real-time prediction of loan approval
- Clean, interactive UI

## ✅ Output

- ✅ Loan is **likely to be accepted**
- ❌ Loan is **likely to be rejected**

## 📦 Future Work

- SHAP Explainability
- Batch Uploads
- Cloud Deployment

---

Made with ❤️ for ML portfolios and real-world use cases.
