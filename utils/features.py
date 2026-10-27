import pandas as pd

CONSTANT_COLS = ["EmployeeCount", "Over18", "StandardHours"]
ID_COLS = ["EmployeeNumber"]
CAT_COLS = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole",
            "MaritalStatus", "OverTime", "Age_Group", "Income_Band", "Tenure_Group",
            "Workload_Category", "Commute_Category", "Promotion_Delay_Category"]
INCOME_LABELS = ["Low", "Lower-Mid", "Upper-Mid", "High"]

def add_features(df, ref):
    """Step 3 features banata hai. ref = original training dataset (bins aur medians ke liye)."""
    df = df.copy()
    edges = ref["MonthlyIncome"].quantile([0, 0.25, 0.5, 0.75, 1]).values
    edges[0], edges[-1] = -float("inf"), float("inf")
    role_median = ref.groupby("JobRole")["MonthlyIncome"].median()

    df["Age_Group"] = pd.cut(df["Age"], bins=[17, 30, 40, 50, 100],
                             labels=["18-30", "31-40", "41-50", "50+"]).astype(str)
    df["Income_Band"] = pd.cut(df["MonthlyIncome"], bins=edges, labels=INCOME_LABELS).astype(str)
    df["Tenure_Group"] = pd.cut(df["YearsAtCompany"], bins=[-1, 2, 5, 10, 100],
                                labels=["New", "Developing", "Established", "Loyal"]).astype(str)
    df["Work_Pressure_Score"] = (df["OverTime"] == "Yes") * 2 + (df["BusinessTravel"] == "Travel_Frequently") * 1
    df["Workload_Category"] = df["Work_Pressure_Score"].map(
        {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}).astype(str)
    df["Commute_Category"] = pd.cut(df["DistanceFromHome"], bins=[0, 5, 15, 100],
                                    labels=["Near", "Medium", "Far"]).astype(str)
    df["Promotion_Delay_Category"] = pd.cut(df["YearsSinceLastPromotion"], bins=[-1, 1, 3, 100],
                                            labels=["Recent", "Moderate", "Long Delay"]).astype(str)
    df["Engagement_Score"] = (df["JobSatisfaction"] + df["WorkLifeBalance"] + df["EnvironmentSatisfaction"]
                              + df["RelationshipSatisfaction"] + df["JobInvolvement"]) / 5
    df["Compensation_Ratio"] = df["MonthlyIncome"] / df["JobRole"].map(role_median)
    df["Career_Growth_Risk"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    return df

def prepare_X(df, ref):
    """Model ke liye X banata hai: features add, IDs/constants drop, categorical ko string."""
    X = add_features(df, ref).drop(columns=CONSTANT_COLS + ID_COLS + ["Attrition", "Attrition_Label"],
                                   errors="ignore")
    X[CAT_COLS] = X[CAT_COLS].astype(str)
    return X
