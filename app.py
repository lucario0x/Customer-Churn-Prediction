import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

MODEL_PATH = "churn_model_pipeline.pkl"
DEFAULT_THRESHOLD = 0.297  # best F2 threshold found in the notebook


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def explain(model, df):
    """Per-feature contributions (log-odds) via XGBoost's built-in pred_contribs. No SHAP needed."""
    pre = model.named_steps["preprocess"]
    booster = model.named_steps["model"].get_booster()
    names = list(pre.get_feature_names_out())
    X = to_dense(pre.transform(df))
    contribs = booster.predict(xgb.DMatrix(X, feature_names=names), pred_contribs=True)
    return pd.Series(contribs[0][:-1], index=names)


def clean_name(n):
    return n.split("__", 1)[-1].replace("_", " = ", 1) if "__" in n else n


model = load_model()

st.title("📉 Customer Churn Predictor")
st.caption("Tuned XGBoost pipeline trained on the Telco Customer Churn dataset.")

with st.sidebar:
    st.header("Decision threshold")
    threshold = st.slider(
        "Flag as 'will churn' above", 0.05, 0.95, DEFAULT_THRESHOLD, 0.01,
        help="The notebook's F2-optimal threshold is 0.297. Lower = catches more churners "
             "but raises more false alarms. 0.5 is the default cutoff.",
    )
    st.markdown("Lower thresholds favour **recall** (catch churners); higher favour **precision**.")

tab_single, tab_batch = st.tabs(["Single customer", "Batch scoring (CSV)"])

# ---------------------------------------------------------------- single customer
with tab_single:
    with st.form("customer"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Profile")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            st.subheader("Services")
            phone = st.selectbox("Phone service", ["Yes", "No"])
            multi = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            yn = ["No", "Yes", "No internet service"]
            sec = st.selectbox("Online security", yn)
            backup = st.selectbox("Online backup", yn)
            device = st.selectbox("Device protection", yn)
            tech = st.selectbox("Tech support", yn)
            tv = st.selectbox("Streaming TV", yn)
            movies = st.selectbox("Streaming movies", yn)

        with c3:
            st.subheader("Billing")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, 0.05)
            total = st.number_input("Total charges ($)", 0.0, 10000.0,
                                    float(round(monthly * max(tenure, 1), 2)), 1.0)

        submitted = st.form_submit_button("Predict churn risk", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
            "OnlineSecurity": sec, "OnlineBackup": backup, "DeviceProtection": device,
            "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total,
        }])

        prob = float(model.predict_proba(row)[0, 1])
        flagged = prob >= threshold

        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{prob:.1%}")
        m2.metric("Decision", "⚠️ Likely to churn" if flagged else "✅ Likely to stay")
        m3.metric("Priority score (prob × monthly $)", f"{prob * monthly:.1f}")
        st.progress(min(max(prob, 0.0), 1.0))

        st.subheader("Why this prediction?")
        contrib = explain(model, row)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)[::-1]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh([clean_name(i) for i in top.index], top.values,
                color=["#d9534f" if v > 0 else "#5cb85c" for v in top.values])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Contribution to churn log-odds (red = raises risk, green = lowers risk)")
        plt.tight_layout()
        st.pyplot(fig)

# ---------------------------------------------------------------- batch
with tab_batch:
    st.markdown("Upload a CSV with the same columns as the Telco dataset (`Churn` and `customerID` optional).")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        data = pd.read_csv(up)
        ids = data["customerID"] if "customerID" in data else pd.Series(range(len(data)), name="row")
        feats = data.drop(columns=[c for c in ["customerID", "Churn"] if c in data], errors="ignore")
        feats["TotalCharges"] = pd.to_numeric(feats["TotalCharges"], errors="coerce")
        feats = feats.fillna({"TotalCharges": 0})

        try:
            probs = model.predict_proba(feats)[:, 1]
        except Exception as e:
            st.error(f"Couldn't score this file. Check that the column names match the training data.\n\n{e}")
            st.stop()

        out = data.copy()
        out["Churn_Probability"] = probs
        out["Flagged"] = probs >= threshold
        out["Priority_Score"] = probs * feats["MonthlyCharges"].values

        k1, k2 = st.columns(2)
        k1.metric("Customers scored", f"{len(out):,}")
        k2.metric("Flagged as likely churners", f"{int(out['Flagged'].sum()):,} ({out['Flagged'].mean():.1%})")

        out = out.sort_values("Priority_Score", ascending=False)
        st.dataframe(out, use_container_width=True, height=420)
        st.download_button("Download scored CSV", out.to_csv(index=False).encode(),
                           "churn_scores.csv", "text/csv")
