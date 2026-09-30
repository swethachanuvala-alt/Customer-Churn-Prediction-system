import datetime as dt

import pandas as pd
import streamlit as st

import churn_logic as L
from train import load_or_train

st.set_page_config(page_title="ChurnGuard | Customer Churn Predictor",
                   page_icon="📉", layout="wide")


def html(s):
    """Render HTML (lines are joined so Markdown never treats indentation as code)."""
    st.markdown(" ".join(line.strip() for line in s.strip().splitlines()),
                unsafe_allow_html=True)


html("""<style>
.block-container {padding-top: 1.5rem; max-width: 1200px;}
.hero {background: linear-gradient(120deg,#6d28d9 0%,#db2777 55%,#f59e0b 100%);
  color:#fff; padding:1.8rem 2.2rem; border-radius:20px; margin-bottom:1.2rem;
  box-shadow:0 10px 30px rgba(109,40,217,.25);}
.hero h1 {margin:0; font-size:2.2rem; color:#fff; letter-spacing:-.5px;}
.hero p {margin:.35rem 0 1rem; opacity:.95; font-size:1.05rem;}
.chips {display:flex; gap:.7rem; flex-wrap:wrap;}
.chip {background:rgba(255,255,255,.2); border:1px solid rgba(255,255,255,.35);
  border-radius:99px; padding:.3rem .9rem; font-weight:600; font-size:.9rem;}
.card-title {font-weight:800; font-size:1.05rem; margin-bottom:.4rem; padding:.35rem .8rem;
  border-radius:10px; display:inline-block;}
.c1 {background:#ede9fe; color:#5b21b6;} .c2 {background:#cffafe; color:#0e7490;}
.c3 {background:#fce7f3; color:#9d174d;} .c4 {background:#fef3c7; color:#92400e;}
.mcard {background:#fff; border-radius:14px; padding:.9rem 1rem; margin-bottom:.7rem;
  border-top:5px solid #7c3aed; box-shadow:0 2px 10px rgba(30,27,75,.08);}
.mcard .v {font-size:1.6rem; font-weight:800; line-height:1.2;}
.mcard .l {font-size:.85rem; color:#6b7280;}
.gauge {background:#fff; border-radius:18px; padding:.8rem; text-align:center;
  box-shadow:0 2px 12px rgba(30,27,75,.10);}
.badge {display:inline-block; color:#fff; font-weight:700; padding:.3rem 1rem;
  border-radius:99px; margin-top:.2rem;}
.b-low {background:#16a34a;} .b-mid {background:#f59e0b;} .b-high {background:#dc2626;}
.verdict {border-radius:14px; padding:1rem 1.2rem; font-size:1rem; line-height:1.5;}
.v-low {background:#dcfce7; color:#14532d;} .v-mid {background:#fef3c7; color:#78350f;}
.v-high {background:#fee2e2; color:#7f1d1d;}
.bar-row {display:flex; align-items:center; gap:.7rem; margin:.4rem 0;}
.bar-label {width:230px; font-size:.92rem; font-weight:500;}
.bar-track {flex:1; background:#ede9fe; border-radius:99px; height:16px; overflow:hidden;}
.bar-fill {height:100%; border-radius:99px;}
.bar-val {width:95px; text-align:right; font-weight:700; font-size:.92rem;}
.tip {background:#ecfeff; border-left:5px solid #06b6d4; border-radius:10px;
  padding:.7rem 1rem; margin:.4rem 0; color:#164e63;}
.stTabs [data-baseweb="tab-list"] {gap:.4rem;}
.stTabs [data-baseweb="tab"] {background:#fff; border-radius:12px 12px 0 0; padding:.6rem 1.2rem;
  font-weight:700;}
.stTabs [aria-selected="true"] {background:#7c3aed !important; color:#fff !important;}
.stButton > button[kind="primary"] {background:linear-gradient(90deg,#7c3aed,#db2777);
  border:none; color:#fff; font-weight:700; border-radius:12px;}
</style>""")


@st.cache_resource
def load():
    b = load_or_train()
    return b["model"], b["columns"], b["metrics"]


@st.cache_data
def dataset():
    return L.load_dataset()


model, columns, metrics = load()
ds = dataset()
importance = L.grouped_importance(model, columns)

# ---------------------------------------------------------------- state / presets
DEFAULTS = dict(senior="No", partner="No", dependents="No", tenure=12, phone="Yes",
                lines="No", internet="Fiber optic", OnlineSecurity="No",
                OnlineBackup="No", DeviceProtection="No", TechSupport="No",
                StreamingTV="No", StreamingMovies="No", contract="Month-to-month",
                paperless="Yes", payment="Electronic check", monthly=70.0)
PRESETS = {
    "🚨 Likely to leave": dict(DEFAULTS, tenure=3, monthly=85.0),
    "😊 Loyal customer": dict(DEFAULTS, partner="Yes", dependents="Yes", tenure=60,
                              lines="Yes", internet="DSL", OnlineSecurity="Yes",
                              OnlineBackup="Yes", DeviceProtection="Yes",
                              TechSupport="Yes", contract="Two year", paperless="No",
                              payment="Credit card (automatic)", monthly=65.0),
    "✍️ Custom": {},
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("preset", "✍️ Custom")
st.session_state.setdefault("history", [])


def apply_preset():
    st.session_state.update(PRESETS.get(st.session_state["preset"], {}))


def gauge_svg(p, color):
    length = 251.2
    return (f'<svg viewBox="0 0 200 122" width="100%">'
            f'<path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#ede9fe" stroke-width="16" stroke-linecap="round"/>'
            f'<path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="{color}" stroke-width="16" stroke-linecap="round" stroke-dasharray="{p * length:.1f} {length}"/>'
            f'<text x="100" y="92" text-anchor="middle" font-size="34" font-weight="800" fill="{color}">{p:.0%}</text>'
            f'<text x="100" y="114" text-anchor="middle" font-size="10" fill="#6b7280">chance this customer leaves</text>'
            f'</svg>')


def hbars(items, color, suffix="%", vmax=None):
    vmax = vmax or max(v for _, v in items) or 1
    rows = "".join(
        f'<div class="bar-row"><div class="bar-label">{lab}</div>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{max(v / vmax * 100, 2):.0f}%;background:{color}"></div></div>'
        f'<div class="bar-val">{v:.1f}{suffix}</div></div>' for lab, v in items)
    html(rows)


BAND_COLOR = {"low": "#16a34a", "mid": "#f59e0b", "high": "#dc2626"}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    threshold = st.slider("Alert threshold", 0.20, 0.80, 0.40, 0.05,
                          help="A customer is flagged 'High risk' at or above this "
                               "probability. Lower = catch more churners, but more false alarms.")
    st.caption("Tip: 0.40 works well because the model misses some churners at 0.50.")
    st.markdown("### 📌 Quick guide")
    st.markdown("1. Pick an example or fill in the customer\n"
                "2. Read the risk score\n3. Try the what-if ideas\n"
                "4. Upload a CSV to score many customers")
    st.markdown("### 🧠 Model")
    st.caption("Random Forest, 200 trees, trained on 7,043 telecom customers.")
    st.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
    st.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2f}")

# ---------------------------------------------------------------- hero
html(f"""<div class="hero"><h1>📉 ChurnGuard</h1>
<p>Customer Churn Prediction System</p>
<p>Spot customers who are about to leave, understand why, and see what could keep them.</p>
<div class="chips"><span class="chip">👥 {len(ds):,} customers analysed</span>
<span class="chip">📉 {ds['Churn'].mean():.1%} overall churn rate</span>
<span class="chip">🎯 {metrics['Accuracy']:.0%} model accuracy</span></div></div>""")

tab_predict, tab_batch, tab_insights, tab_about = st.tabs(
    ["🔮 Predict", "📁 Batch upload", "📊 Insights", "ℹ️ Help & model"])

# ================================================================ TAB 1: PREDICT
with tab_predict:
    st.radio("Start from an example", list(PRESETS), key="preset", horizontal=True,
             on_change=apply_preset)

    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        html('<div class="card-title c1">👤 About the customer</div>')
        st.selectbox("Senior citizen", ["No", "Yes"], key="senior")
        st.selectbox("Has a partner", ["No", "Yes"], key="partner")
        st.selectbox("Has dependents", ["No", "Yes"], key="dependents")
        st.slider("Tenure (months with us)", 0, 72, key="tenure")

    with c2, st.container(border=True):
        html('<div class="card-title c2">📡 Services</div>')
        st.selectbox("Phone service", ["Yes", "No"], key="phone")
        if st.session_state["phone"] == "Yes":
            st.selectbox("Multiple lines", ["No", "Yes"], key="lines")
            lines = st.session_state["lines"]
        else:
            lines = "No phone service"
        st.selectbox("Internet service", ["Fiber optic", "DSL", "No"], key="internet")
        addons = {}
        if st.session_state["internet"] == "No":
            addons = {k: "No internet service" for k in L.ADDONS}
            st.caption("No internet, so add-ons don't apply.")
        else:
            for k in L.ADDONS:
                st.selectbox(L.FRIENDLY[k], ["No", "Yes"], key=k)
                addons[k] = st.session_state[k]

    with c3, st.container(border=True):
        html('<div class="card-title c3">💳 Billing</div>')
        st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], key="contract")
        st.selectbox("Paperless billing", ["Yes", "No"], key="paperless")
        st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                        "Bank transfer (automatic)",
                                        "Credit card (automatic)"], key="payment")
        st.number_input("Monthly charges ($)", 15.0, 130.0, step=0.5, key="monthly")
        total = st.session_state["tenure"] * st.session_state["monthly"]
        st.caption(f"Estimated total paid so far: ${total:,.0f}")

    S = st.session_state
    row = {"SeniorCitizen": 1 if S["senior"] == "Yes" else 0, "Partner": S["partner"],
           "Dependents": S["dependents"], "tenure": S["tenure"],
           "PhoneService": S["phone"], "MultipleLines": lines,
           "InternetService": S["internet"], **addons, "Contract": S["contract"],
           "PaperlessBilling": S["paperless"], "PaymentMethod": S["payment"],
           "MonthlyCharges": float(S["monthly"]), "TotalCharges": float(total)}

    p = L.predict_one(model, columns, row)
    band, label = L.risk_band(p, threshold)
    color = BAND_COLOR[band]
    scenarios = [w for w in L.what_if(model, columns, row, p) if w["delta"] < -0.01]

    st.markdown("### 🎯 Result")
    r1, r2, r3 = st.columns([1.1, 1, 1.4])
    with r1:
        html(f'<div class="gauge">{gauge_svg(p, color)}'
             f'<span class="badge b-{band}">{label}</span></div>')
    with r2:
        stage = ("🌱 New" if S["tenure"] < 6 else "🌿 Growing" if S["tenure"] < 24
                 else "🌳 Established" if S["tenure"] < 48 else "🏆 Veteran")
        html(f'<div class="mcard" style="border-top-color:#db2777"><div class="v">${S["monthly"] * 12 * p:,.0f}</div>'
             f'<div class="l">expected yearly revenue at risk</div></div>')
        html(f'<div class="mcard" style="border-top-color:#06b6d4"><div class="v">${S["monthly"] * 12:,.0f}</div>'
             f'<div class="l">yearly value of this customer</div></div>')
        html(f'<div class="mcard" style="border-top-color:#f59e0b"><div class="v">{stage}</div>'
             f'<div class="l">customer stage ({S["tenure"]} months)</div></div>')
    with r3:
        msg = {"high": "⚠️ <b>This customer is likely to leave.</b> Reach out soon with one of the offers below.",
               "mid": "👀 <b>Worth keeping an eye on.</b> A small nudge now could prevent a cancellation.",
               "low": "✅ <b>This customer looks stable.</b> No urgent action needed. Keep the service quality up."}[band]
        html(f'<div class="verdict v-{band}">{msg}</div>')
        if band != "low" and scenarios:
            st.markdown("**💡 Best ways to keep them**")
            for w in scenarios[:3]:
                html(f'<div class="tip"><b>{w["label"]}</b><br>'
                     f'Estimated risk {p:.0%} → {w["new_p"]:.0%} '
                     f'(<b>{-w["delta"] * 100:.0f} points lower</b>)</div>')
        elif band != "low":
            st.info("No single change lowers this customer's risk much. Consider a personal call.")

    if scenarios:
        with st.expander("🧪 What-if simulator (all ideas)", expanded=(band != "low")):
            st.caption("Estimated by the model, not guaranteed. Only ideas that lower the risk are shown.")
            hbars([(w["label"], w["new_p"] * 100) for w in scenarios],
                  "linear-gradient(90deg,#06b6d4,#7c3aed)", "% risk", vmax=100)

    b1, b2 = st.columns([1, 3])
    with b1:
        if st.button("💾 Save to history", type="primary"):
            st.session_state.history.append({
                "Time": dt.datetime.now().strftime("%H:%M:%S"),
                "Tenure": S["tenure"], "Contract": S["contract"],
                "Internet": S["internet"], "Monthly $": S["monthly"],
                "Churn probability (%)": round(p * 100, 1), "Risk": label})
            st.toast("Saved!", icon="✅")
    if st.session_state.history:
        with st.expander(f"🗂️ Prediction history ({len(st.session_state.history)})"):
            hist = pd.DataFrame(st.session_state.history)
            st.dataframe(hist, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download history (CSV)", hist.to_csv(index=False),
                               "prediction_history.csv", "text/csv")

# ================================================================ TAB 2: BATCH
with tab_batch:
    st.markdown("### 📁 Score many customers at once")
    st.write("Upload a CSV with the same columns as the training data "
             "(`customerID` is optional, `Churn` is ignored). You get a ranked list "
             "with the riskiest customers on top.")
    template = pd.read_csv("customer_churn_dataset.csv").drop(columns="Churn").head(10)
    st.download_button("⬇️ Download a sample template", template.to_csv(index=False),
                       "churn_template.csv", "text/csv")
    up = st.file_uploader("Upload your CSV", type="csv")
    if up is not None:
        try:
            res = L.predict_batch(model, columns, pd.read_csv(up), threshold)
            n_high = int((res["Risk"] == "High risk").sum())
            m1, m2, m3 = st.columns(3)
            m1.metric("Customers", f"{len(res):,}")
            m2.metric("High risk", f"{n_high:,}")
            m3.metric("Average churn probability", f"{res['Churn probability (%)'].mean():.1f}%")
            st.dataframe(res, use_container_width=True, hide_index=True,
                         column_config={"Churn probability (%)": st.column_config.ProgressColumn(
                             "Churn probability (%)", min_value=0, max_value=100, format="%.0f")})
            st.download_button("⬇️ Download results (CSV)", res.to_csv(index=False),
                               "churn_predictions.csv", "text/csv", type="primary")
        except Exception as e:
            st.error(f"Could not score this file. {e}")

# ================================================================ TAB 3: INSIGHTS
with tab_insights:
    st.markdown("### 📊 What drives churn in this data?")
    tables = L.insight_tables(ds)
    ct, tn, it, pm = (tables["Contract type"], tables["Tenure"],
                      tables["Internet type"], tables["Payment method"])
    st.success(
        f"**Key takeaways:** month-to-month customers churn at {ct['Month-to-month']:.0f}% "
        f"versus {ct['Two year']:.0f}% on two-year contracts. Customers in their first year "
        f"churn at {tn['0-12 months']:.0f}%. Electronic-check payers churn at "
        f"{pm['Electronic check']:.0f}% and fiber-optic users at {it['Fiber optic']:.0f}%.")
    palette = ["#7c3aed", "#db2777", "#06b6d4", "#f59e0b", "#16a34a", "#ef4444"]
    ia, ib = st.columns(2)
    for i, (name, ser) in enumerate(tables.items()):
        with (ia if i % 2 == 0 else ib), st.container(border=True):
            st.markdown(f"**Churn rate by {name.lower()}**")
            hbars(list(ser.items()), palette[i % 6], "%", vmax=60)
    with st.container(border=True):
        st.markdown("**🌲 What the model pays most attention to**")
        st.caption("Share of the Random Forest's decisions that rely on each feature.")
        hbars([(k, v) for k, v in importance.head(10).items()],
              "linear-gradient(90deg,#7c3aed,#db2777)", "%")

# ================================================================ TAB 4: HELP
with tab_about:
    st.markdown("### 🧠 How good is the model?")
    cols = st.columns(5)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.1%}" if k != "ROC-AUC" else f"{v:.2f}")
    st.info("**Being honest about limits:** at a 50% cut-off the model finds roughly half of "
            "the customers who really leave (recall). That's why the default alert threshold "
            "is 40%. Use scores to prioritise outreach, not as certainty.")
    with st.expander("📖 What do these numbers mean?", expanded=True):
        st.markdown(
            "- **Accuracy**: share of all predictions that were right.\n"
            "- **Precision**: of customers flagged as leaving, how many really left.\n"
            "- **Recall**: of customers who really left, how many we caught.\n"
            "- **F1**: a balance of precision and recall.\n"
            "- **ROC-AUC**: how well the model ranks risky customers above safe ones "
            "(0.5 = guessing, 1.0 = perfect).")
    with st.expander("🚦 What do the risk levels mean?"):
        st.markdown(
            f"- 🟢 **Low**: below {max(threshold - 0.15, 0.10):.0%}\n"
            f"- 🟡 **Medium**: {max(threshold - 0.15, 0.10):.0%} to {threshold:.0%}\n"
            f"- 🔴 **High**: {threshold:.0%} and above\n\n"
            "Change the alert threshold in the sidebar.")
    with st.expander("🧪 How does the what-if simulator work?"):
        st.write("It changes one thing about the customer (for example the contract), asks the "
                 "model for a new score, and shows the difference. It reflects patterns in past "
                 "data, so treat it as a guide, not a promise.")
    with st.expander("❓ Troubleshooting"):
        st.markdown("- Model error on start: run `python train.py` to rebuild `churn_model.pkl`.\n"
                    "- Batch upload error: download the template and match its column names.")
    st.caption("Built with Streamlit and scikit-learn. Trained on the Telco customer churn dataset.")
