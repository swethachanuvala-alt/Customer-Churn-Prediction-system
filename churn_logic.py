"""All the non-UI logic: predictions, what-if scenarios, insights."""
import pandas as pd

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

FRIENDLY = {
    "SeniorCitizen": "Senior citizen", "Partner": "Has partner",
    "Dependents": "Has dependents", "tenure": "Tenure (months)",
    "PhoneService": "Phone service", "MultipleLines": "Multiple lines",
    "InternetService": "Internet type", "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "TechSupport": "Tech support", "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies", "Contract": "Contract type",
    "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges",
}


def load_dataset(path="customer_churn_dataset.csv"):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df


def predict_one(model, columns, row):
    return float(model.predict_proba(pd.DataFrame([row])[columns])[0, 1])


def risk_band(p, threshold):
    if p >= threshold:
        return "high", "High risk"
    if p >= max(threshold - 0.15, 0.10):
        return "mid", "Medium risk"
    return "low", "Low risk"


def _apply(row, changes):
    r = dict(row)
    r.update(changes)
    if "MonthlyCharges" in changes:
        r["TotalCharges"] = r["tenure"] * r["MonthlyCharges"]
    return r


def what_if(model, columns, row, base_p):
    """Try common retention actions and see how the churn probability would change."""
    has_net = row["InternetService"] != "No"
    manual_pay = row["PaymentMethod"] in ("Electronic check", "Mailed check")
    sc = []
    if row["Contract"] == "Month-to-month":
        sc.append(("Move to a 1-year contract", {"Contract": "One year"}))
    if row["Contract"] != "Two year":
        sc.append(("Move to a 2-year contract", {"Contract": "Two year"}))
    if manual_pay:
        sc.append(("Switch to automatic card payment",
                   {"PaymentMethod": "Credit card (automatic)"}))
    if has_net and row["TechSupport"] == "No":
        sc.append(("Add tech support", {"TechSupport": "Yes"}))
    if has_net and row["OnlineSecurity"] == "No":
        sc.append(("Add online security", {"OnlineSecurity": "Yes"}))
    sc.append(("Give a 10% price discount",
               {"MonthlyCharges": round(row["MonthlyCharges"] * 0.9, 2)}))

    pkg = {}
    if row["Contract"] != "Two year":
        pkg["Contract"] = "Two year"
    if manual_pay:
        pkg["PaymentMethod"] = "Credit card (automatic)"
    if has_net:
        for k in ("TechSupport", "OnlineSecurity"):
            if row[k] == "No":
                pkg[k] = "Yes"
    if len(pkg) >= 2:
        sc.append(("Full retention package (all of the above)", pkg))

    out = []
    for label, changes in sc:
        new_p = predict_one(model, columns, _apply(row, changes))
        out.append({"label": label, "new_p": new_p, "delta": new_p - base_p})
    return sorted(out, key=lambda d: d["new_p"])


def grouped_importance(model, columns):
    """Random Forest importance, summed back to the original (un-one-hot) features, in %."""
    names = model.named_steps["prep"].get_feature_names_out()
    imp = model.named_steps["rf"].feature_importances_
    out = {c: 0.0 for c in columns}
    for n, v in zip(names, imp):
        raw = n.split("__", 1)[1]
        for c in sorted(columns, key=len, reverse=True):
            if raw == c or raw.startswith(c + "_"):
                out[c] += float(v)
                break
    s = pd.Series(out).sort_values(ascending=False) * 100
    s.index = [FRIENDLY.get(i, i) for i in s.index]
    return s


def churn_rate_by(df, col):
    return (df.groupby(col, observed=True)["Churn"].mean() * 100).round(1)


def insight_tables(df):
    d = df.copy()
    d["Tenure group"] = pd.cut(d["tenure"], [-1, 12, 24, 48, 72],
                               labels=["0-12 months", "13-24 months",
                                       "25-48 months", "49-72 months"])
    d["Monthly bill"] = pd.cut(d["MonthlyCharges"], [0, 35, 60, 85, 200],
                               labels=["Under $35", "$35 to $60", "$60 to $85", "Over $85"])
    d["Senior"] = d["SeniorCitizen"].map({0: "Not senior", 1: "Senior citizen"})
    return {
        "Contract type": churn_rate_by(d, "Contract"),
        "Tenure": churn_rate_by(d, "Tenure group"),
        "Internet type": churn_rate_by(d, "InternetService"),
        "Payment method": churn_rate_by(d, "PaymentMethod"),
        "Tech support": churn_rate_by(d[d["InternetService"] != "No"], "TechSupport"),
        "Monthly bill": churn_rate_by(d, "Monthly bill"),
    }


def predict_batch(model, columns, df, threshold):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError("Your file is missing these columns: " + ", ".join(missing))
    X = df[columns].copy()
    for c in NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    prob = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["Churn probability (%)"] = (prob * 100).round(1)
    out["Risk"] = [risk_band(p, threshold)[1] for p in prob]
    return out.sort_values("Churn probability (%)", ascending=False).reset_index(drop=True)
