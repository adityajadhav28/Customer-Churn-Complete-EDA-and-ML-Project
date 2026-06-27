import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-card .label { font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }
    .metric-card .value { font-size: 24px; font-weight: 700; color: #00d4ff; }
    .churn-yes {
        background: #450a0a; border: 1px solid #dc2626;
        border-radius: 10px; padding: 20px; text-align: center;
        color: #fca5a5; font-size: 20px; font-weight: 700;
    }
    .churn-no {
        background: #052e16; border: 1px solid #16a34a;
        border-radius: 10px; padding: 20px; text-align: center;
        color: #86efac; font-size: 20px; font-weight: 700;
    }
    .section-header {
        font-size: 13px; font-weight: 600; color: #00d4ff;
        text-transform: uppercase; letter-spacing: 0.1em;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 6px; margin-bottom: 12px; margin-top: 18px;
    }
    div[data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
    .stButton > button {
        background: #00d4ff; color: #000; font-weight: 700;
        border: none; border-radius: 8px; width: 100%;
        padding: 14px; font-size: 16px; margin-top: 10px;
    }
    .stButton > button:hover { background: #00b8d9; color: #000; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load("churn_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    scaler       = joblib.load("scaler.pkl")
    return model, preprocessor, scaler

try:
    model, preprocessor, scaler = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("##  Customer Churn Prediction")
st.markdown("**XGBoost model** · Telco customer data · Real-time churn risk prediction")
st.markdown("---")

if not model_loaded:
    st.error(f"⚠️ Could not load model files. Make sure `churn_model.pkl`, `preprocessor.pkl`, and `scaler.pkl` are in the same folder.\n\n`{load_error}`")
    st.stop()

# ─────────────────────────────────────────────
# Sidebar — Input Form
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("###  Customer Details")

    st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
    gender      = st.selectbox("Gender", ["Male", "Female"])
    senior      = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner     = st.selectbox("Partner", ["Yes", "No"])
    dependents  = st.selectbox("Dependents", ["Yes", "No"])

    st.markdown('<div class="section-header">Account</div>', unsafe_allow_html=True)
    tenure      = st.slider("Tenure (months)", 0, 72, 12)
    contract    = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless   = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment     = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly     = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5)
    total       = st.number_input("Total Charges ($)",   min_value=0.0, value=float(tenure * monthly), step=1.0)

    st.markdown('<div class="section-header">Phone Services</div>', unsafe_allow_html=True)
    phone       = st.selectbox("Phone Service", ["Yes", "No"])
    multiple    = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    st.markdown('<div class="section-header">Internet Services</div>', unsafe_allow_html=True)
    internet    = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security    = st.selectbox("Online Security",   ["Yes", "No", "No internet service"])
    backup      = st.selectbox("Online Backup",     ["Yes", "No", "No internet service"])
    protection  = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    support     = st.selectbox("Tech Support",      ["Yes", "No", "No internet service"])
    tv          = st.selectbox("Streaming TV",      ["Yes", "No", "No internet service"])
    movies      = st.selectbox("Streaming Movies",  ["Yes", "No", "No internet service"])

    predict_btn = st.button("🔍 Predict Churn")

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
if predict_btn:
    input_data = pd.DataFrame([{
        "customerID":        "0000-TEST",
        "gender":            gender,
        "SeniorCitizen":     senior,
        "Partner":           partner,
        "Dependents":        dependents,
        "tenure":            tenure,
        "PhoneService":      phone,
        "MultipleLines":     multiple,
        "InternetService":   internet,
        "OnlineSecurity":    security,
        "OnlineBackup":      backup,
        "DeviceProtection":  protection,
        "TechSupport":       support,
        "StreamingTV":       tv,
        "StreamingMovies":   movies,
        "Contract":          contract,
        "PaperlessBilling":  paperless,
        "PaymentMethod":     payment,
        "MonthlyCharges":    monthly,
        "TotalCharges":      str(total),
    }])

    try:
        X_encoded  = preprocessor.transform(input_data)
        X_scaled   = scaler.transform(X_encoded)
        prediction  = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        churn_prob  = probability[1]
        stay_prob   = probability[0]
        success = True
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        success = False

    if success:
        col_result, col_metrics, col_chart = st.columns([2, 1.5, 2])

        with col_result:
            st.markdown("### Prediction Result")
            if prediction == 1:
                st.markdown(
                    f'<div class="churn-yes">⚠️ Likely to Churn'
                    f'<br><span style="font-size:14px;font-weight:400;">Churn probability: {churn_prob*100:.1f}%</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="churn-no">✅ Likely to Stay'
                    f'<br><span style="font-size:14px;font-weight:400;">Retention probability: {stay_prob*100:.1f}%</span></div>',
                    unsafe_allow_html=True
                )

            st.markdown("#### Risk Level")
            if churn_prob < 0.3:
                risk_color = "#16a34a"; risk_label = "🟢 Low Risk"
            elif churn_prob < 0.6:
                risk_color = "#d97706"; risk_label = "🟡 Medium Risk"
            else:
                risk_color = "#dc2626"; risk_label = "🔴 High Risk"
            st.markdown(f'<p style="color:{risk_color};font-size:18px;font-weight:700;">{risk_label}</p>', unsafe_allow_html=True)
            st.progress(float(churn_prob))

        with col_metrics:
            st.markdown("### Key Inputs")
            st.markdown(f'<div class="metric-card"><div class="label">Tenure</div><div class="value">{tenure}mo</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="label">Monthly</div><div class="value">${monthly:.0f}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="label">Contract</div><div class="value" style="font-size:14px;">{contract}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="label">Internet</div><div class="value" style="font-size:14px;">{internet}</div></div>', unsafe_allow_html=True)

        with col_chart:
            st.markdown("### Probability Breakdown")
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor("#0f172a")
            ax.set_facecolor("#0f172a")
            wedges, texts, autotexts = ax.pie(
                [stay_prob, churn_prob],
                labels=["Stay", "Churn"],
                autopct="%1.1f%%",
                colors=["#16a34a", "#dc2626"],
                startangle=90,
                textprops={"color": "#e2e8f0", "fontsize": 12}
            )
            for at in autotexts:
                at.set_color("white"); at.set_fontweight("bold")
            ax.set_title("Stay vs Churn", color="#e2e8f0", fontsize=12, pad=10)
            st.pyplot(fig)
            plt.close(fig)

        # Business Insights
        st.markdown("---")
        st.markdown("### 💡 Business Insights")
        insights = []
        if contract == "Month-to-month":
            insights.append("🔴 **Month-to-month contract** — highest churn segment. Offer a discount to upgrade to annual.")
        if internet == "Fiber optic" and churn_prob > 0.4:
            insights.append("🔴 **Fiber optic customer at risk** — fiber users churn more despite premium service. Review pricing.")
        if tenure < 12:
            insights.append("🟡 **New customer (< 12 months)** — early churn risk is high. Trigger an onboarding check-in.")
        if security == "No" or support == "No":
            insights.append("🟡 **Missing value-added services** — customers without security/support churn more. Offer a free trial.")
        if payment == "Electronic check":
            insights.append("🟡 **Electronic check payment** — correlates with higher churn. Encourage auto-pay enrollment.")
        if churn_prob < 0.25:
            insights.append("🟢 **Low churn risk** — satisfied customer. Good candidate for upsell or referral program.")
        if not insights:
            insights.append("📊 No strong churn signals detected. Monitor billing and support interactions.")
        for i in insights:
            st.markdown(f"- {i}")

else:
    # Landing state
    st.markdown("### 👈 Fill in customer details in the sidebar and click **Predict Churn**")
    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><div class="label">Model</div><div class="value" style="font-size:18px;">XGBoost</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><div class="label">Best F1-Score</div><div class="value">55.70%</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><div class="label">Recall</div><div class="value">51.74%</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><div class="label">Dataset</div><div class="value" style="font-size:18px;">Telco</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Model Comparison")
    st.dataframe(pd.DataFrame({
        "Model":      ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost ✓"],
        "Accuracy":   ["79.13%", "77.43%", "79.21%", "78.21%"],
        "Precision":  ["71.58%", "58.68%", "65.87%", "60.31%"],
        "Recall":     ["35.12%", "49.87%", "44.50%", "51.74%"],
        "F1-Score":   ["47.12%", "53.91%", "53.12%", "55.70%"],
    }), use_container_width=True, hide_index=True)
    st.caption("XGBoost selected as final model — highest Recall and F1-Score, best at catching at-risk customers.")
