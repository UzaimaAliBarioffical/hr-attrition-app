import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from utils.features import add_features, prepare_X

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "dataset.csv")
MODEL_PATH = os.path.join(BASE, "models", "final_pipeline.pkl")

st.set_page_config(page_title="HR Attrition Intelligence", layout="wide")


@st.cache_data
def load_raw():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def prepared():
    raw = load_raw()
    feat = add_features(raw, raw)
    feat["Attrition"] = raw["Attrition"].map({"No": 0, "Yes": 1})
    X = prepare_X(raw, raw)
    y = raw["Attrition"].map({"No": 0, "Yes": 1})
    return feat, X, y


@st.cache_data
def split_data():
    _, X, y = prepared()
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def baseline_row(raw):
    row = {}
    for c in raw.columns:
        if c == "Attrition":
            continue
        row[c] = raw[c].mode().iloc[0] if raw[c].dtype == object else raw[c].median()
    return pd.DataFrame([row])


def rate_series(feat, col):
    return (feat.groupby(col, observed=True)["Attrition"].mean() * 100).round(1).sort_values(ascending=False)


# ---------------- Pages ----------------

def page_home():
    raw = load_raw()
    st.title("HR Workforce Analytics & Employee Attrition Intelligence")
    st.markdown("""
**Business problem:** Organization mein employee attrition ho raha hai. HR yeh samajhna chahta hai ki
kaun se employees zyada chhod rahe hain, kaun se factors (overtime, salary, promotion delay, travel)
attrition se juде hain, aur kaun si interventions retention badha sakti hain.

**Objectives:** Data quality check, workforce EDA, compensation aur engagement analysis,
classification models ka comparison, aur ek risk assessment tool jo sirf voluntary retention support ke liye hai.
    """)
    st.info("Ethical use: Yeh prediction sirf workforce-level analysis aur voluntary retention support ke liye hai. "
            "Ise termination, promotion, salary, disciplinary ya hiring decisions ka akela basis na banayein.")
    st.subheader("Dataset summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{raw.shape[0]:,}")
    c2.metric("Columns", raw.shape[1])
    c3.metric("Attrition rate", f"{(raw['Attrition'] == 'Yes').mean() * 100:.1f}%")

    st.subheader("HR Recommendations")
    recs = [
        "**Overtime kam karein:** Overtime attrition se juda hai. Workload planning aur overtime limits lagayein. (Attrition Analysis page dekhein)",
        "**Work-life balance improve karein:** Low WLB score wale employees ke liye flexible schedule options dein. (Employee Engagement page)",
        "**Compensation review:** Income band ke hisaab se salary benchmarks check karein, khaaskar niche wale bands mein. (Compensation Analysis page)",
        "**Career growth:** Lambe samay se promotion na mile wale employees ke liye career review aur clear promotion path banayein.",
        "**Manager training:** Manager rating aur engagement ke relation ko dekhte hue managers ko feedback aur coaching training dein.",
        "**Travel reduction:** Frequent business travel wale employees ke liye travel policy review karein.",
        "**Onboarding aur engagement programs:** Naye employees ke liye structured onboarding aur regular check-ins rakhein.",
    ]
    for r in recs:
        st.markdown(f"- {r}")


def page_quality():
    raw = load_raw()
    st.title("Data Quality Report")
    missing = raw.isna().sum()
    st.subheader("Missing values")
    st.write(f"Total missing values: **{int(missing.sum())}**")
    st.dataframe(pd.DataFrame({"Missing": missing, "Percent": (missing / len(raw) * 100).round(2)}))
    st.subheader("Duplicates")
    st.write(f"Duplicate rows: **{int(raw.duplicated().sum())}**")
    st.subheader("Data types")
    st.dataframe(pd.DataFrame({"Type": raw.dtypes.astype(str)}))
    st.subheader("Numeric summary")
    st.dataframe(raw.describe().T.round(2))
    st.subheader("Outliers (IQR rule, sirf report, delete nahi kiye)")
    num = raw.select_dtypes("number")
    q1, q3 = num.quantile(0.25), num.quantile(0.75)
    iqr = q3 - q1
    out = ((num < q1 - 1.5 * iqr) | (num > q3 + 1.5 * iqr)).sum()
    st.dataframe(pd.DataFrame({"Outliers": out[out > 0]}))
    st.subheader("Categorical values")
    for col in raw.select_dtypes(include="object").columns:
        st.write(f"**{col}:** {', '.join(map(str, raw[col].unique()))}")


def page_exec():
    feat, _, _ = prepared()
    st.title("HR Executive Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total employees", f"{len(feat):,}")
    c2.metric("Attrition count", int(feat["Attrition"].sum()))
    c3.metric("Attrition rate", f"{feat['Attrition'].mean() * 100:.1f}%")
    c4.metric("Avg monthly income", f"{feat['MonthlyIncome'].mean():,.0f}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg tenure (yrs)", f"{feat['YearsAtCompany'].mean():.1f}")
    c2.metric("Avg job satisfaction", f"{feat['JobSatisfaction'].mean():.2f}")
    c3.metric("Avg work-life balance", f"{feat['WorkLifeBalance'].mean():.2f}")
    c4.metric("Overtime %", f"{(feat['OverTime'] == 'Yes').mean() * 100:.1f}%")
    c1, c2 = st.columns(2)
    c1.metric("Promotion-eligible (3+ yrs since promotion)", int((feat["YearsSinceLastPromotion"] >= 3).sum()))
    c2.metric("Avg trainings last year", f"{feat['TrainingTimesLastYear'].mean():.1f}")
    st.caption("Note: dataset mein absence days aur manager rating columns nahi hain, isliye unki jagah related columns use hue hain.")
    st.subheader("Attrition rate by department (%)")
    st.bar_chart(rate_series(feat, "Department"))


def page_demographics():
    feat, _, _ = prepared()
    st.title("Workforce Demographics")
    depts = st.multiselect("Department filter", sorted(feat["Department"].unique()),
                           default=sorted(feat["Department"].unique()))
    f = feat[feat["Department"].isin(depts)]
    st.write(f"Filtered employees: **{len(f):,}**")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Gender")
        st.bar_chart(f["Gender"].value_counts())
        st.subheader("Age group")
        st.bar_chart(f["Age_Group"].value_counts().sort_index())
    with c2:
        st.subheader("Job level")
        st.bar_chart(f["JobLevel"].value_counts().sort_index())
        st.subheader("Attrition rate by gender (%)")
        st.bar_chart(rate_series(f, "Gender"))


def page_compensation():
    feat, _, _ = prepared()
    st.title("Compensation Analysis")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Avg monthly income by department")
        st.bar_chart(feat.groupby("Department")["MonthlyIncome"].mean().round(0))
        st.subheader("Avg monthly income by job level")
        st.bar_chart(feat.groupby("JobLevel")["MonthlyIncome"].mean().round(0))
    with c2:
        st.subheader("Avg monthly income by job role")
        st.bar_chart(feat.groupby("JobRole")["MonthlyIncome"].mean().round(0).sort_values())
        st.subheader("Attrition rate by stock option level (%)")
        st.bar_chart(rate_series(feat, "StockOptionLevel").sort_index())
    st.subheader("Income vs attrition (median)")
    st.dataframe(feat.groupby("Attrition")[["MonthlyIncome", "PercentSalaryHike", "StockOptionLevel"]].median().rename(
        index={0: "Stayed", 1: "Left"}))
    st.subheader("Attrition rate by income band (%)")
    st.bar_chart(rate_series(feat, "Income_Band").reindex(["Low", "Lower-Mid", "Upper-Mid", "High"]))


def page_engagement():
    feat, _, _ = prepared()
    st.title("Employee Engagement")
    cols = ["JobSatisfaction", "WorkLifeBalance", "EnvironmentSatisfaction",
            "RelationshipSatisfaction", "JobInvolvement", "PerformanceRating", "TrainingTimesLastYear"]
    st.subheader("Average score: stayed vs left")
    st.dataframe(feat.groupby("Attrition")[cols].mean().T.rename(columns={0: "Stayed", 1: "Left"}).round(2))
    st.subheader("Attrition rate by work-life balance (%)")
    st.bar_chart(rate_series(feat, "WorkLifeBalance").sort_index())
    st.subheader("Attrition rate by job satisfaction (%)")
    st.bar_chart(rate_series(feat, "JobSatisfaction").sort_index())
    st.subheader("Attrition rate by environment satisfaction (%)")
    st.bar_chart(rate_series(feat, "EnvironmentSatisfaction").sort_index())
    st.subheader("Engagement score (average) by attrition")
    st.bar_chart(feat.groupby("Attrition")["Engagement_Score"].mean().rename(index={0: "Stayed", 1: "Left"}))


def page_attrition():
    feat, _, _ = prepared()
    st.title("Attrition Analysis")
    options = ["OverTime", "BusinessTravel", "Department", "JobRole", "Promotion_Delay_Category",
               "Income_Band", "Tenure_Group", "Workload_Category", "Commute_Category", "Gender", "MaritalStatus"]
    col = st.selectbox("Attrition rate (%) kis column se dekhna hai?", options)
    st.bar_chart(rate_series(feat, col))
    st.subheader("Overall")
    c1, c2, c3 = st.columns(3)
    c1.metric("Overtime attrition", f"{feat[feat['OverTime'] == 'Yes']['Attrition'].mean() * 100:.1f}%")
    c2.metric("No overtime attrition", f"{feat[feat['OverTime'] == 'No']['Attrition'].mean() * 100:.1f}%")
    c3.metric("New employees (<=2 yrs)", f"{feat[feat['Tenure_Group'] == 'New']['Attrition'].mean() * 100:.1f}%")


def page_preprocessing():
    raw = load_raw()
    model = load_model()
    Xtr, Xte, ytr, yte = split_data()
    st.title("Preprocessing Laboratory")
    num_cnt = Xtr.select_dtypes("number").shape[1]
    cat_cnt = Xtr.select_dtypes(exclude="number").shape[1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Numeric features", num_cnt)
    c2.metric("Categorical features", cat_cnt)
    c3.metric("Processed features (after encoding)", len(model.named_steps["prep"].get_feature_names_out()))
    c1, c2 = st.columns(2)
    c1.metric("Train shape", str(Xtr.shape))
    c2.metric("Test shape", str(Xte.shape))
    st.subheader("Missing values in raw data")
    st.write(f"Total missing: **{int(raw.isna().sum().sum())}** (dataset clean hai, isliye koi imputation real data par zaroori nahi tha)")
    st.subheader("Pipeline steps")
    st.markdown("""
- **Numeric:** median imputation, then standard scaling
- **Categorical:** most-frequent imputation, then one-hot encoding (handle_unknown='ignore')
- **Split:** 80/20 stratified, random_state=42
- Preprocessing sirf training data par fit hoti hai (data leakage se bachav)
    """)
    st.subheader("Missing-value strategy comparison (notebook result)")
    st.write("Median (normalized MAE sabse kam tha) > Mean > Most-frequent. Row removal se 91% data kho jaata tha, isliye woh reject hua.")


def page_models():
    model = load_model()
    st.title("Model Comparison")
    results = pd.DataFrame([
        {"Model": "Logistic Regression", "CV ROC-AUC": 0.823, "Accuracy": 0.776, "Precision": 0.388, "Recall": 0.702, "F1": 0.500, "Test ROC-AUC": 0.839},
        {"Model": "Random Forest", "CV ROC-AUC": 0.829, "Accuracy": 0.847, "Precision": 0.625, "Recall": 0.106, "F1": 0.182, "Test ROC-AUC": 0.805},
        {"Model": "Gradient Boosting", "CV ROC-AUC": 0.825, "Accuracy": 0.857, "Precision": 0.632, "Recall": 0.255, "F1": 0.364, "Test ROC-AUC": 0.793},
        {"Model": "Decision Tree", "CV ROC-AUC": 0.695, "Accuracy": 0.738, "Precision": 0.292, "Recall": 0.447, "F1": 0.353, "Test ROC-AUC": 0.556},
    ]).set_index("Model")
    st.caption("Yeh table notebook ke training run ke results hain.")
    st.dataframe(results)
    st.info("Logistic Regression ko final model chuna gaya kyunki attrition ke case mein Recall sabse important hai (0.70), aur test ROC-AUC bhi sabse achha hai.")

    st.subheader("Top 15 features (Logistic Regression coefficients)")
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["model"].coef_[0]
    s = pd.Series(coefs, index=names)
    top = s.reindex(s.abs().sort_values(ascending=False).index).head(15).sort_values()
    st.bar_chart(top)
    st.caption("Positive coefficient = attrition badhata hai. Negative = attrition kam karta hai.")


def page_risk():
    raw = load_raw()
    model = load_model()
    st.title("Attrition Risk Assessment")
    st.warning("Yeh tool sirf voluntary retention support ke liye hai. Ise termination, promotion, salary ya disciplinary decisions ka akela basis na banayein.")
    base = baseline_row(raw)
    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.slider("Age", 18, 60, int(base["Age"].iloc[0]))
        income = c2.number_input("Monthly income", 1000, 20000, int(base["MonthlyIncome"].iloc[0]), step=100)
        distance = c3.slider("Distance from home (km)", 1, 30, int(base["DistanceFromHome"].iloc[0]))
        c1, c2, c3 = st.columns(3)
        years = c1.slider("Years at company", 0, 40, int(base["YearsAtCompany"].iloc[0]))
        since_promo = c2.slider("Years since last promotion", 0, 15, int(base["YearsSinceLastPromotion"].iloc[0]))
        overtime = c3.selectbox("OverTime", ["No", "Yes"], index=0)
        c1, c2, c3 = st.columns(3)
        travel = c1.selectbox("Business travel", sorted(raw["BusinessTravel"].unique()))
        dept = c2.selectbox("Department", sorted(raw["Department"].unique()))
        role = c3.selectbox("Job role", sorted(raw["JobRole"].unique()))
        c1, c2, c3 = st.columns(3)
        job_sat = c1.slider("Job satisfaction (1-4)", 1, 4, int(base["JobSatisfaction"].iloc[0]))
        wlb = c2.slider("Work-life balance (1-4)", 1, 4, int(base["WorkLifeBalance"].iloc[0]))
        env_sat = c3.slider("Environment satisfaction (1-4)", 1, 4, int(base["EnvironmentSatisfaction"].iloc[0]))
        submitted = st.form_submit_button("Calculate risk")

    if submitted:
        row = base.copy()
        row.loc[0, "Age"] = age
        row.loc[0, "MonthlyIncome"] = income
        row.loc[0, "DistanceFromHome"] = distance
        row.loc[0, "YearsAtCompany"] = years
        row.loc[0, "YearsSinceLastPromotion"] = since_promo
        row.loc[0, "OverTime"] = overtime
        row.loc[0, "BusinessTravel"] = travel
        row.loc[0, "Department"] = dept
        row.loc[0, "JobRole"] = role
        row.loc[0, "JobSatisfaction"] = job_sat
        row.loc[0, "WorkLifeBalance"] = wlb
        row.loc[0, "EnvironmentSatisfaction"] = env_sat

        X_row = prepare_X(row, raw)
        prob = float(model.predict_proba(X_row)[:, 1][0])
        level = "Low" if prob < 0.3 else "Medium" if prob < 0.5 else "High" if prob < 0.7 else "Critical"

        st.subheader("Result")
        c1, c2 = st.columns(2)
        c1.metric("Attrition probability", f"{prob * 100:.1f}%")
        c2.metric("Risk level", level)
        st.caption("Note: probability model ka score hai, exact chance nahi. Thresholds sirf guide ke liye hain.")

        st.subheader("Main risk indicators")
        indicators = []
        if overtime == "Yes":
            indicators.append("Overtime kar raha hai")
        if travel == "Travel_Frequently":
            indicators.append("Frequent business travel")
        if since_promo >= 3:
            indicators.append("Promotion mein deri (3+ saal)")
        if income < raw["MonthlyIncome"].median():
            indicators.append("Income median se kam")
        if job_sat <= 2:
            indicators.append("Job satisfaction kam")
        if wlb <= 2:
            indicators.append("Work-life balance kam")
        st.write(", ".join(indicators) if indicators else "Koi major risk indicator nahi mila.")

        st.subheader("Suggested HR interventions (voluntary support)")
        actions = []
        if overtime == "Yes":
            actions.append("Overtime aur workload review karein")
        if since_promo >= 3:
            actions.append("Career growth discussion aur promotion path")
        if income < raw["MonthlyIncome"].median():
            actions.append("Compensation review")
        if job_sat <= 2 or wlb <= 2:
            actions.append("Engagement check-in aur flexible schedule options")
        if travel == "Travel_Frequently":
            actions.append("Travel reduction ya flexibility")
        for a in (actions or ["Regular engagement check-in"]):
            st.markdown(f"- {a}")


def page_responsible():
    model = load_model()
    _, Xte, _, yte = split_data()
    st.title("Responsible ML Review")
    st.info("This prediction is intended for workforce-level analysis and voluntary retention support. "
            "It must not be used as the sole basis for termination, promotion, salary, disciplinary, or hiring decisions.")
    pred = model.predict(Xte)
    yt_all = np.asarray(yte)
    rows = []
    for col in ["Gender", "Age_Group", "Department", "BusinessTravel"]:
        vals = Xte[col].values
        for g in sorted(pd.unique(vals)):
            m = vals == g
            yt, yp = yt_all[m], pred[m]
            tp = int(((yt == 1) & (yp == 1)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            tn = int(((yt == 0) & (yp == 0)).sum())
            rows.append({
                "Group": f"{col}: {g}",
                "N (test)": int(m.sum()),
                "Actual attrition %": round(100 * yt.mean(), 1),
                "Predicted attrition %": round(100 * yp.mean(), 1),
                "Recall %": round(100 * tp / (tp + fn), 1) if (tp + fn) else None,
                "False positive rate %": round(100 * fp / (fp + tn), 1) if (fp + tn) else None,
            })
    st.subheader("Group-wise performance (test set)")
    st.dataframe(pd.DataFrame(rows))
    st.caption("Chhote groups mein numbers unstable hote hain. Large differences ko aur data se verify karein.")


# ---------------- Router ----------------

PAGES = {
    "Home": page_home,
    "Data Quality Report": page_quality,
    "HR Executive Dashboard": page_exec,
    "Workforce Demographics": page_demographics,
    "Compensation Analysis": page_compensation,
    "Employee Engagement": page_engagement,
    "Attrition Analysis": page_attrition,
    "Preprocessing Laboratory": page_preprocessing,
    "Model Comparison": page_models,
    "Attrition Risk Assessment": page_risk,
    "Responsible ML Review": page_responsible,
}

choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
PAGES[choice]()
