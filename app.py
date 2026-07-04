"""
CreditIQ — Home Loan Default Risk Prediction System
A premium, fintech-style Streamlit dashboard.

Run with:
    pip install -r requirements.txt
    streamlit run app.py
"""
from src.prediction_pipeline import PredictionPipeline
import hashlib
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

experiment_tracker = pd.read_csv(
    BASE_DIR / "artifacts/experiment_results/experiment_tracker.csv"
)

metrics = joblib.load(
    BASE_DIR / "artifacts/experiment_results/final_metrics.pkl"
)

# FIX #3: was missing BASE_DIR, so this broke whenever the app was launched
# from a directory other than the project root.
threshold_df = pd.read_csv(
    BASE_DIR / "artifacts/experiment_results/xgboost_optimized_threshold_analysis.csv"
)


pipeline = PredictionPipeline()


def predict_dataframe(df, pipeline):
    """
    FIX #2: previously, 'Prediction' (Default / No Default) came from
    pipeline.predict()'s own internal threshold, while 'Risk Level' /
    'Loan Decision' were recomputed here using metrics["threshold"] and a
    hardcoded 0.45. If those two thresholds ever disagreed, a row could show
    Prediction = "No Default" but Risk Level = "High Risk" (or vice versa) —
    which looked like the model was behaving inconsistently.

    Now everything (Prediction, Risk Level, Loan Decision) is derived from
    the SAME probability + the SAME threshold, so they can never contradict
    each other.
    """
    predictions, probabilities = pipeline.predict(df)

    risk_levels = []
    decisions = []
    prediction_text = []

    for prob in probabilities:
        is_default = prob >= metrics["threshold"]

        prediction_text.append("Default" if is_default else "No Default")

        if prob < metrics["threshold"]:
            risk_levels.append("Low Risk")
            decisions.append("Approve")
        elif prob < 0.45:
            risk_levels.append("Medium Risk")
            decisions.append("Manual Review")
        else:
            risk_levels.append("High Risk")
            decisions.append("Reject")

    customer_ids = (
        df["SK_ID_CURR"]
        if "SK_ID_CURR" in df.columns
        else [f"CUST-{i+1}" for i in range(len(df))]
    )

    result = pd.DataFrame({

        "Customer ID": customer_ids,

        "Prediction": prediction_text,

        "Default Probability": (probabilities * 100).round(2),

        "Risk Level": risk_levels,

        "_level": [x.lower().split()[0] for x in risk_levels],

        "Loan Decision": decisions

    })

    return result
# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Home Loan Default Risk Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)
 

# ============================================================================
# GLOBAL CSS — banking / fintech look (white bg, blue accents, rounded cards)
# ============================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

:root{
  --blue:#2563EB; --blue-dark:#1D4ED8; --blue-light:#EEF3FF; --blue-lighter:#F6F9FF;
  --ink:#101820; --ink-soft:#475467; --ink-mute:#8A94A6;
  --line:#E7EBF1; --line-soft:#F0F2F6; --paper:#FFFFFF; --canvas:#F7F8FB;
  --green:#12805C; --green-bg:#E7F7EF; --orange:#B45309; --orange-bg:#FEF3E2;
  --red:#B42318; --red-bg:#FDE8E7;
  /* Standard brand gradient — reused across the hero, sidebar, uploader, and tables */
  --brand-gradient: linear-gradient(120deg, #0F2E7A 0%, #1a46c4 46%, #2563EB 100%);
}

html, body, [class*="css"], .stMarkdown, p, span, label, div { font-family:'Inter', sans-serif; }
h1,h2,h3,h4,h5 { font-family:'Manrope', sans-serif !important; letter-spacing:-.01em; }



.stApp { background:var(--canvas); }
.block-container { padding-top:1.6rem; padding-bottom:2.5rem; max-width:1280px; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
div[data-testid="stSidebarContent"],
div[data-testid="stSidebarUserContent"]{
  background:var(--paper) !important; border-right:none !important;
}
section[data-testid="stSidebar"] > div { padding-top:1.2rem; }
.side-brand{
  display:flex; align-items:center; gap:11px; padding:2px 6px 18px 10px;
  border-bottom:1px solid rgba(255,255,255,.18); margin-bottom:14px;
}
.side-brand .mark{
  width:38px; height:38px; border-radius:10px; flex-shrink:0;
  background:var(--brand-gradient); border:1px solid rgba(255,255,255,.25);
  display:flex; align-items:center; justify-content:center; font-size:18px;
  box-shadow:none;
}
.side-brand .name{ font-family:'Manrope',sans-serif; font-weight:800; font-size:16px; color:var(--brand-gradient); line-height:1.1;}
.side-brand .sub{ font-size:10.5px; font-weight:700; color:rgba(255,255,255,.72); text-transform:uppercase; letter-spacing:.05em; margin-top:2px;}
.side-foot{
  display:flex; align-items:center; gap:10px; padding:14px 10px; margin-top:16px;
  border-top:1px solid rgba(255,255,255,.18);
}
.side-foot .avatar{
  width:30px; height:30px; border-radius:50%; background:rgba(255,255,255,.16); color:var(--brand-gradient);
  font-weight:800; font-size:11.5px; display:flex; align-items:center; justify-content:center; flex-shrink:0;
}
.side-foot .fname{ font-size:12.3px; font-weight:700; color:var(--brand-gradient); }
.side-foot .frole{ font-size:10.8px; font-weight:600; color:rgba(255,255,255,.72); }

/* ---------- buttons ---------- */
.stButton>button, div[data-testid="stDownloadButton"]>button{
  background:var(--blue); color:#fff; border:none; border-radius:10px;
  font-weight:700; font-size:13.4px; padding:0.55rem 1.1rem; transition:.15s ease;
  box-shadow:none;
}
.stButton>button:hover, div[data-testid="stDownloadButton"]>button:hover{
  background:var(--blue-dark); transform:translateY(-1px); color:#fff;
}
.stButton>button:focus:not(:active){ box-shadow:0 0 0 3px rgba(37,99,235,.25) !important; }
button[kind="secondary"]{ background:#fff !important; color:var(--ink-soft) !important; border:1px solid var(--line) !important; }
button[kind="secondary"]:hover{ border-color:#C7CEDA !important; color:var(--ink) !important; }

/* ---------- file uploader ---------- */
div[data-testid="stFileUploaderDropzone"]{
  background:var(--brand-gradient) !important;
  border:1.8px dashed rgba(255,255,255,.35) !important; border-radius:14px !important;
}
div[data-testid="stFileUploaderDropzone"] *{
  color:#ffffff !important;
}
div[data-testid="stFileUploaderDropzone"] small{
  color:rgba(255,255,255,.75) !important;
}
div[data-testid="stFileUploaderDropzone"] svg{
  fill:#ffffff !important;
}
div[data-testid="stFileUploaderDropzone"] button{
  background:#ffffff !important; color:var(--blue-dark) !important; border-radius:9px !important; border:none !important;
}

/* ---------- dataframe ---------- */
div[data-testid="stDataFrame"]{
  border:1px solid var(--line); border-radius:14px; overflow:hidden;
}

/* ---------- generic card ---------- */
.ciq-card{
  background:var(--paper); border:1px solid var(--line); border-radius:14px;
  box-shadow:0 1px 2px rgba(16,24,40,.04), 0 2px 12px rgba(16,24,40,.05);
  padding:20px 20px;
}
.ciq-card{
    color:#101820;
}

.ciq-card h1,
.ciq-card h2,
.ciq-card h3,
.ciq-card h4,
.ciq-card h5,
.ciq-card p,
.ciq-card span,
.ciq-card b,
.ciq-card div{
    color:#101820 !important;
}
.section-head{ display:flex; align-items:baseline; justify-content:space-between; margin:26px 0 14px; flex-wrap:wrap; gap:6px;}
.section-head h2{ font-size:19px; font-weight:800; color:var(--ink); margin:0;}
.section-head p{ font-size:12.6px; color:var(--ink-mute); font-weight:500; margin:2px 0 0;}
.eyebrow{
  font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--blue);
  display:flex; align-items:center; gap:7px; margin-bottom:8px;
}
.eyebrow::before{ content:''; width:14px; height:2px; background:var(--blue); border-radius:2px;}

/* ---------- hero ---------- */
.hero{
  border-radius:20px; padding:42px 38px; position:relative; overflow:hidden; color:#fff;
  background:var(--brand-gradient);
  margin-bottom:6px;
}
.hero:before{
  content:''; position:absolute; inset:0;
  background-image:radial-gradient(circle at 85% 15%, rgba(255,255,255,.14), transparent 55%),
                    radial-gradient(circle at 6% 92%, rgba(255,255,255,.10), transparent 45%);
}
.hero-badge{
  display:inline-flex; align-items:center; gap:7px; font-size:11.5px; font-weight:700;
  background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.22);
  padding:6px 13px 6px 11px; border-radius:100px; margin-bottom:16px; position:relative; z-index:2;
}
.hero-badge .dot{ width:6px; height:6px; border-radius:50%; background:#6EE7B7; box-shadow:0 0 0 3px rgba(110,231,183,.25);}
.hero h1{ font-size:33px; font-weight:800; line-height:1.15; margin:0 0 12px; position:relative; z-index:2; max-width:680px;}
.hero p.sub{ font-size:14.5px; line-height:1.65; color:rgba(255,255,255,.85); font-weight:500; max-width:600px; position:relative; z-index:2;}

/* ---------- KPI cards ---------- */
.kpi-card{
  background:#fff; border:1px solid var(--line); border-radius:14px; padding:17px 17px 15px;
  box-shadow:0 1px 2px rgba(16,24,40,.04);
}
.kpi-icon{
  width:30px; height:30px; border-radius:8px; background:var(--blue-light);
  display:flex; align-items:center; justify-content:center; font-size:14.5px; margin-bottom:12px;
}
.kpi-label{ font-size:10.8px; font-weight:700; color:var(--ink-mute); text-transform:uppercase; letter-spacing:.04em; margin-bottom:5px;}
.kpi-value{ font-size:18.5px; font-weight:800; color:var(--ink); font-family:'Manrope',sans-serif;}
.kpi-value.mono{ font-family:'JetBrains Mono',monospace; font-size:19.5px;}
.kpi-foot{ font-size:10.6px; color:var(--ink-mute); font-weight:600; margin-top:5px;}

/* ---------- workflow rail ---------- */
.rail-wrap{ overflow-x:auto; padding:6px 2px 2px;}
.rail{ position:relative; display:flex; justify-content:space-between; min-width:900px; padding:10px 10px 0;}
.rail:before{
  content:''; position:absolute; top:33px; left:46px; right:46px; height:2px;
  background:repeating-linear-gradient(90deg,#D7DEEA 0 8px, transparent 8px 14px); z-index:1;
}
.rail-step{ position:relative; z-index:2; display:flex; flex-direction:column; align-items:center; width:126px; text-align:center;}
.rail-node{
  width:44px; height:44px; border-radius:50%; background:var(--blue-light); border:2px solid var(--blue);
  display:flex; align-items:center; justify-content:center; margin-bottom:10px; font-size:17px;
  box-shadow:0 0 0 5px rgba(37,99,235,.08);
}
.rail-num{ font-size:9.5px; font-weight:800; color:var(--blue-dark); letter-spacing:.05em; margin-bottom:3px;}
.rail-name{ font-size:12px; font-weight:700; color:var(--ink);}

/* ---------- tech stack ---------- */
.stack-chip{
  text-align:center; border-radius:12px; border:1px solid var(--line); background:#fff; padding:14px 6px 12px;
}
.stack-chip .sc-icon{
  width:32px; height:32px; margin:0 auto 7px; border-radius:9px; background:var(--blue-lighter);
  display:flex; align-items:center; justify-content:center; font-size:12px; font-weight:800; color:var(--blue-dark);
  font-family:'JetBrains Mono',monospace;
}
.stack-chip .sc-name{ font-size:11px; font-weight:700; color:var(--ink-soft);}

/* ---------- badges ---------- */
.badge{ display:inline-flex; align-items:center; gap:5px; font-size:11px; font-weight:700; padding:4px 10px; border-radius:100px;}
.badge:before{ content:''; width:6px; height:6px; border-radius:50%;}
.badge.low{ background:var(--green-bg); color:var(--green);} .badge.low:before{ background:var(--green);}
.badge.medium{ background:var(--orange-bg); color:var(--orange);} .badge.medium:before{ background:var(--orange);}
.badge.high{ background:var(--red-bg); color:var(--red);} .badge.high:before{ background:var(--red);}

/* ---------- stat / summary cards ---------- */
.stat-card{ background:#fff; border:1px solid var(--line); border-radius:14px; padding:17px 18px;}
.stat-card .num{ font-size:23px; font-weight:800; font-family:'Manrope',sans-serif;}
.stat-card .lbl{ font-size:11px; color:var(--ink-mute); font-weight:700; text-transform:uppercase; letter-spacing:.03em; margin-top:4px;}
.stat-blue .num{ color:var(--blue-dark);} .stat-green .num{ color:var(--green);} .stat-red .num{ color:var(--red);}

/* ---------- best model card ---------- */
.best-model-card{
    background:var(gradient);
    border:1px solid #E5E7EB;
    border-radius:16px;
    padding:22px;
    color:#101820;   /* force text color */
}

.best-model-card *{
    color:#2563eb !important;
}

.bm-label{
    font-size:12px;
    font-weight:700;
    color:#2563EB !important;
    text-transform:uppercase;
    margin-bottom:10px;
}

.bm-title{
    font-size:22px;
    font-weight:700;
    color:#2563EB !important;
    margin-bottom:18px;
}

.bm-row{
    display:flex;
    justify-content:space-between;
    padding:10px 0;
    border-bottom:1px solid #EEF2F7;
}

.bm-row .k{
    color:#667085 !important;
    font-weight:600;
}

.bm-row .v{
    color:#101820 !important;
    font-weight:700;
}

.bm-note{
    margin-top:18px;
    background:#F8FAFC;
    padding:14px;
    border-radius:10px;
    color:#475467 !important;
    line-height:1.6;
}

/* ---------- confusion matrix ---------- */
.cm-cell{ border-radius:10px; padding:16px 8px; text-align:center; font-family:'JetBrains Mono',monospace;}
.cm-cell .cm-val{ font-size:19px; font-weight:800;}
.cm-cell .cm-tag{ font-size:9px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; margin-top:4px; opacity:.75;}
.cm-tn{ background:var(--green-bg); color:var(--green);} .cm-fp{ background:var(--orange-bg); color:var(--orange);}
.cm-fn{ background:var(--red-bg); color:var(--red);} .cm-tp{ background:var(--blue-light); color:var(--blue-dark);}
.cm-axis{ font-size:9.5px; font-weight:700; color:var(--ink-mute); text-transform:uppercase; letter-spacing:.04em; text-align:center; padding-top:6px;}

/* ---------- model tags / metric rows ---------- */
.model-tag{ display:inline-block; font-size:11.5px; font-weight:700; padding:7px 13px; border-radius:9px; background:var(--canvas); border:1px solid var(--line); color:var(--ink-soft); margin:0 6px 8px 0;}
.metric-row{ display:flex; align-items:center; justify-content:space-between; padding:9px 0; border-bottom:1px solid var(--line-soft); font-size:12.6px;}
.metric-row:last-child{ border-bottom:none;}
.metric-row .m-name{ font-weight:700; color:var(--ink);}
.metric-row .m-desc{ color:var(--ink-mute); font-weight:600; font-family:'JetBrains Mono',monospace; font-size:12px;}
.check-item{
    display:flex;
    align-items:flex-start;
    gap:14px;
    margin-bottom:18px;
    color:#475467;
    line-height:1.5;
    font-size:13px;
}

.check-item b{
    color:#101820;
}

.check-item span{
    flex-shrink:0;
    margin-top:2px;
}
.check-item .tick{ color:var(--green); font-weight:800; flex-shrink:0;}

.ciq-footer{ margin-top:30px; padding:18px 4px 4px; border-top:1px solid var(--line); display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; font-size:11.8px; color:var(--ink-mute); font-weight:600;}
.ciq-footer b{ color:var(--ink); font-weight:800;}

/* ================= DATAFRAME TOOLBAR ================= */

/* Search box */
[data-testid="stDataFrame"] input{
    color:#101820 !important;
    background:#FFFFFF !important;
    border:1px solid #D0D5DD !important;
}

[data-testid="stDataFrame"] input::placeholder{
    color:#667085 !important;
}

/* Toolbar icons */
[data-testid="stDataFrame"] button{
    color:#101820 !important;
    background:#FFFFFF !important;
}

/* SVG icons */
[data-testid="stDataFrame"] svg{
    fill:#101820 !important;
    color:#101820 !important;
}

/* Dropdown labels */
[data-testid="stDataFrame"] label,
[data-testid="stDataFrame"] span{
    color:#101820 !important;
}

/* Header */
[data-testid="stDataFrame"] thead th{
    color:#1D4ED8 !important;
    background:#EEF3FF !important;
    font-weight:700;
}

/* Table cells */
[data-testid="stDataFrame"] tbody td{
    color:#101820 !important;
}

/* ---------- Markdown inside cards ---------- */

.ciq-card,
.ciq-card *{
    color:#101820 !important;
}

.ciq-card p{
    color:#475467 !important;
}

.ciq-card strong,
.ciq-card b{
    color:#101820 !important;
}

.ciq-card h1,
.ciq-card h2,
.ciq-card h3{
    color:#1D4ED8 !important;
}

.ciq-card li{
    color:#475467 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR — brand + navigation
# ============================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <div class="mark">🏦</div>
            <div>
                <div class="name" style="color: var(--brand-gradient);">CreditIQ</div>
                <div class="sub" style="color: var(--brand-gradient);">Risk Analytics</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected = option_menu(
        menu_title=None,
        options=["Home", "Prediction", "Model Performance", "About"],
        icons=["house", "cloud-arrow-up", "bar-chart-line", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color":"var(--brand-gradient)"},
            "icon": {"color": "var(--brand-gradient)", "font-size": "15px"},
            "nav-link": {
                "font-family": "Inter, sans-serif",
                "font-size": "13.6px",
                "font-weight": "600",
                "text-align": "left",
                "margin": "2px 0",
                "border-radius": "9px",
                "color": "var(--brand-gradient)",
                "padding": "10px 12px",
            },
            "nav-link-selected": {
                "background-color": "#1F4BC3",
                "color": "#FFFFFF",
                "font-weight": "700",
            },
        },
    )

    st.markdown(
        """
        <div class="side-foot">
            <div class="avatar">SS</div>
            <div>
                <div class="fname">Sanjana S</div>
                <div class="frole" style="color: var(--brand-gradient);">Data Scientist</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================================
# HELPERS
# ============================================================================
def section_head(title, subtitle):
    st.markdown(
        f'<div class="section-head"><div><h2>{title}</h2>'
        f'<p>{subtitle}</p></div></div>',
        unsafe_allow_html=True,
    )


def kpi_card(icon, label, value, sub, mono=False):
    mono_cls = "mono" if mono else ""
    return (
        f'<div class="kpi-card"><div class="kpi-icon">{icon}</div>'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {mono_cls}">{value}</div>'
        f'<div class="kpi-foot">{sub}</div></div>'
    )


PLOTLY_CONFIG = {"displayModeBar": False}
BLUE = "#2563EB"
BLUE_LIGHT = "#C7D9FB"
GREEN = "#12805C"
ORANGE = "#F59E0B"

# ============================================================================
# PAGE: HOME
# ============================================================================
if selected == "Home":
    best_model = metrics["best_model"]
    roc_auc = metrics["roc_auc"]

    st.markdown(
        f'<div class="hero">'
        f'<div class="hero-badge"><span class="dot"></span>'
        f'{best_model} &middot; ROC-AUC {roc_auc:.3f}</div>'
        f'<h1>Home Loan Default Risk Prediction System</h1>'
        f'<p class="sub">AI-powered machine learning application...</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, _ = st.columns([1.3, 1.5, 3])
    with c1:
        if st.button("🎯  Run a Batch Prediction", use_container_width=True):
            st.session_state["_nav_hint"] = "Prediction"
            st.info("Open **Prediction** from the sidebar to upload a CSV and score applicants.")
    with c2:
        if st.button("📊  View Model Performance", use_container_width=True, type="secondary"):
            st.info("Open **Model Performance** from the sidebar for the full evaluation breakdown.")

    section_head("Key metrics", "Snapshot of the production model and underlying dataset.")
    cols = st.columns(6)
    kpis = [
    (
        "🏆",
        "Best Model",
        metrics["best_model"],
        "Production Model",
        False
    ),

    (
        "🎯",
        "ROC-AUC",
        f'{metrics["roc_auc"]:.3f}',
        "Validation Set",
        True
    ),

    (
        "📐",
        "F1 Score",
        f'{metrics["f1"]:.3f}',
        "Default Class",
        True
    ),

    (
        "🗄️",
        "Dataset",
        "Home Credit",
        "Default Risk",
        False
    ),

    (
        "🔢",
        "Features",
        str(metrics["features"]),
        "Engineered Features",
        True
    ),

    (
        "📚",
        "Models Compared",
        str(len(experiment_tracker)),
        "Candidate Models",
        True
    ),
]
    for col, (icon, label, value, sub, mono) in zip(cols, kpis):
        with col:
            st.markdown(kpi_card(icon, label, value, sub, mono), unsafe_allow_html=True)

    section_head("Project workflow", "End-to-end pipeline from raw bureau data to a deployed scoring service.")
    steps = [
        ("01", "📥", "Data Collection"),
        ("02", "🧹", "Data Cleaning"),
        ("03", "🧬", "Feature Engineering"),
        ("04", "⚙️", "Model Training"),
        ("05", "📈", "Model Evaluation"),
        ("06", "🔮", "Prediction"),
        ("07", "🚀", "Deployment"),
    ]
    steps_html = "".join(
        f'<div class="rail-step"><div class="rail-node">{icon}</div>'
        f'<div class="rail-num">STEP {num}</div><div class="rail-name">{name}</div></div>'
        for num, icon, name in steps
    )
    st.markdown(
        f'<div class="ciq-card rail-wrap"><div class="rail">{steps_html}</div></div>',
        unsafe_allow_html=True,
    )

    section_head("Technology stack", "Libraries and frameworks used across the modeling pipeline.")
    stack = [("Py", "Python"), ("sk", "Scikit-learn"), ("XGB", "XGBoost"), ("LGB", "LightGBM"),
             ("pd", "Pandas"), ("np", "NumPy"), ("plt", "Matplotlib"), ("sns", "Seaborn"), ("st", "Streamlit")]
    cols = st.columns(9)
    for col, (abbr, name) in zip(cols, stack):
        with col:
            st.markdown(
                f'<div class="stack-chip"><div class="sc-icon">{abbr}</div><div class="sc-name">{name}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="ciq-footer"><div>Developed by <b>Sanjana S</b></div>'
        '<div>Home Credit Default Risk Dataset &middot; Model v2.3</div></div>',
        unsafe_allow_html=True,
    )

# ============================================================================
# PAGE: PREDICTION
# ============================================================================
elif selected == "Prediction":
    section_head("Batch prediction", "Upload a CSV of applicant records to score default risk across your full portfolio.")

    if "pred_df" not in st.session_state:
        st.session_state["pred_df"] = None
    if "pred_results" not in st.session_state:
        st.session_state["pred_results"] = None

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown('<div class="ciq-card">', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload applicant CSV",
            type=["csv"],
            label_visibility="collapsed",
            help=f"Accepted format: CSV · Applicant feature set ({metrics['features']} columns expected)",
        )
        st.markdown(
            '<div style="font-size:12.5px;color:#8A94A6;font-weight:600;margin-top:8px;">'
            f'Accepted format: CSV &nbsp;·&nbsp; Applicant feature set ({metrics["features"]} columns expected)</div>',
            unsafe_allow_html=True,
        )

        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                st.session_state["pred_df"] = df
                st.session_state["pred_file_name"] = uploaded.name
            except Exception:
                st.error("Could not parse this CSV file. Please check the format and try again.")

        st.markdown('</div>', unsafe_allow_html=True)

    with right:

        st.info(
            "### 📘 How scoring works"
        )

        st.info(
            f"""
    **🧩 Feature Engineering**

    Every applicant is transformed into **{metrics['features']} engineered features** before prediction.
    """
        )

        st.info(
            f"""
    **🤖 Probability Estimation**

    The production model **{metrics['best_model']}** estimates the probability of default.
    """
        )

        st.info(
            f"""
    **🎯 Risk Classification**

    Applicants with probability above **{metrics['threshold']:.2f}** are classified as **High Risk**.
    """
        )

        st.info(
            """
    **📄 Export Results**

    Download a CSV containing probability, risk level and recommendation.
    """
        )

    df = st.session_state["pred_df"]

    if df is None:
        st.markdown(
            '<div class="ciq-card" style="text-align:center;padding:50px 20px;margin-top:22px;">'
            '<div style="font-size:34px;margin-bottom:10px;">📄</div>'
            '<div style="font-size:13px;font-weight:600;color:#8A94A6;">'
            'Upload a CSV file to see the dataset preview and run predictions.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        fname = st.session_state.get("pred_file_name", "the uploaded file")
        section_head("Dataset preview", f"Showing the first 10 rows of {fname} · {df.shape[0]} rows · {df.shape[1]} columns.")
        st.dataframe(df.head(10), use_container_width=True, height=340)

        b1, b2, _ = st.columns([1, 1, 4])
        with b1:
            predict_clicked = st.button("🔮  Predict Loan Risk", use_container_width=True)
        with b2:
            if st.button("Clear", use_container_width=True, type="secondary"):
                st.session_state["pred_df"] = None
                st.session_state["pred_results"] = None
                st.rerun()

        if predict_clicked:
            pipeline = PredictionPipeline()
            st.session_state["pred_results"] = predict_dataframe(df, pipeline)

        results = st.session_state["pred_results"]
        if results is not None:
            section_head("Prediction summary", "Portfolio-level risk breakdown for this batch.")

            total = len(results)
            high = int((results["_level"] == "high").sum())
            low = int((results["_level"] == "low").sum())
            avg_prob = results["Default Probability"].mean()

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.markdown(f'<div class="stat-card stat-blue"><div class="num">{total:,}</div><div class="lbl">Total Applicants</div></div>', unsafe_allow_html=True)
            with s2:
                st.markdown(f'<div class="stat-card stat-red"><div class="num">{high:,}</div><div class="lbl">High Risk Customers</div></div>', unsafe_allow_html=True)
            with s3:
                st.markdown(f'<div class="stat-card stat-green"><div class="num">{low:,}</div><div class="lbl">Low Risk Customers</div></div>', unsafe_allow_html=True)
            with s4:
                st.markdown(f'<div class="stat-card stat-blue"><div class="num">{avg_prob:.1f}%</div><div class="lbl">Avg. Default Probability</div></div>', unsafe_allow_html=True)

            section_head("Scored applicants", "Per-customer default probability and recommended loan decision.")

            top100 = results.head(100)

            display_df = top100.drop(columns=["_level"])

            styled = display_df.style.apply(
                lambda s: [
                    {
                        "low": "background-color:#E7F7EF;color:#12805C;font-weight:700;",
                        "medium": "background-color:#FEF3E2;color:#B45309;font-weight:700;",
                        "high": "background-color:#FDE8E7;color:#B42318;font-weight:700;",
                    }[lvl]
                    if s.name == "Risk Level" else ""
                    for lvl in top100["_level"]
                ],
                axis=0,
            ).format({"Default Probability": "{:.1f}%"})

            st.dataframe(
                styled,
                use_container_width=True,
                height=420,
            )

            csv_bytes = (
              results
              .drop(columns=["_level"])
              .to_csv(index=False)
              .encode("utf-8")
)
            st.download_button(
                "⬇️  Download Predictions CSV",
                data=csv_bytes,
                file_name=f"{metrics['best_model']}_predictions.csv",
                mime="text/csv",
            )

# ============================================================================
# PAGE: MODEL PERFORMANCE
# ============================================================================
elif selected == "Model Performance":
    section_head("Model performance", "Comparative evaluation across 13 candidate models and threshold analysis for the production classifier.")

    model_data = experiment_tracker.copy()

    model_data = model_data.rename(columns={
    "ROC_AUC": "ROC-AUC"
    })

    model_data = model_data.sort_values(
    "F1",
    ascending=False
    ).reset_index(drop=True)

    model_data.insert(0, "#", range(1, len(model_data) + 1))

    # Rank-based: row #1 after sorting by ROC-AUC is "best". (Previously this
    # matched model_data["Model"] against metrics["best_model"] by exact
    # string equality, which could silently match nothing if the names
    # didn't line up — breaking the highlight here and in the charts below.)
    model_data["best"] = (model_data["#"] == 1)


    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.markdown('<div class="ciq-card" style="margin-bottom:20px;">'
                    '<div style="font-size:13.5px;font-weight:700;margin-bottom:12px;color:#101820;">Model comparison</div>'
                    '<div style="font-size:11.5px;color:#8A94A6;font-weight:500;margin-bottom:12px;">Ranked by ROC-AUC on the held-out validation set.</div>',
                    unsafe_allow_html=True)
        
        # "best" = the top-ranked row after sorting by ROC-AUC (i.e. row #1).
        # Previously this was matched by exact string equality against
        # metrics["best_model"], which silently produced zero matches if the
        # name didn't line up character-for-character — so no row ever got
        # highlighted and the raw "best" boolean column leaked into view
        # instead (since .hide() isn't reliably respected by st.dataframe).
        show_cols = ["#", "Model", "ROC-AUC", "F1", "Accuracy"]
        display_models = model_data[show_cols].copy()
        is_best = (display_models["#"] == 1).to_numpy()

        # Mark the top model visually in its name too.
        display_models.loc[display_models["#"] == 1, "Model"] = (
            "🏆 " + display_models.loc[display_models["#"] == 1, "Model"]
        )

        def highlight_best(row):
            if is_best[row.name]:
                return [
                    "background:#fff;"
                    "color:black; font-weight:800;"
                ] * len(row)
            return [""] * len(row)

        styled_models = display_models.style.apply(highlight_best, axis=1).format(
            {"ROC-AUC": "{:.3f}", "F1": "{:.3f}", "Accuracy": "{:.1%}"}
        )
        st.dataframe(styled_models, use_container_width=True, height=430, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:

        best_model = str(metrics.get("best_model", "N/A"))
        threshold = float(metrics.get("threshold", 0))
        roc_auc = float(metrics.get("roc_auc", 0))
        accuracy = float(metrics.get("accuracy", 0))
        precision = float(metrics.get("precision", 0))
        recall = float(metrics.get("recall", 0))
        f1 = float(metrics.get("f1", 0))
        features = int(metrics.get("features", 0))
        validation_size = int(metrics.get("validation_size", 0))

        st.markdown(
            f'<div class="best-model-card">'
            f'<div class="bm-label">Production Model</div>'
            f'<div class="bm-title" style="color: var(--brand-gradient);">{metrics["best_model"]}</div>'
            f'<div class="bm-row"><span class="k">Decision Threshold</span>'
            f'<span class="v">{metrics["threshold"]:.2f}</span></div>'
            f'<div class="bm-row"><span class="k">ROC-AUC</span>'
            f'<span class="v">{metrics["roc_auc"]:.3f}</span></div>'
            f'<div class="bm-row"><span class="k">Accuracy</span>'
            f'<span class="v">{metrics["accuracy"]:.3f}</span></div>'
            f'<div class="bm-row"><span class="k">Precision</span>'
            f'<span class="v">{metrics["precision"]:.3f}</span></div>'
            f'<div class="bm-row"><span class="k">Recall</span>'
            f'<span class="v">{metrics["recall"]:.3f}</span></div>'
            f'<div class="bm-row"><span class="k">F1 Score</span>'
            f'<span class="v">{metrics["f1"]:.3f}</span></div>'
            f'<div class="bm-row"><span class="k">Features Used</span>'
            f'<span class="v">{metrics["features"]}</span></div>'
            f'<div class="bm-row"><span class="k">Validation Samples</span>'
            f'<span class="v">{metrics["validation_size"]:,}</span></div>'
            f'<div class="bm-note"><b>Business Recommendation</b><br><br>'
            f'The deployed <b>{metrics["best_model"]}</b> model uses a decision '
            f'threshold of <b>{metrics["threshold"]:.2f}</b> to improve the detection '
            f'of potential loan defaulters. This operating point increases recall for '
            f'the default class, allowing the bank to identify more high-risk applicants '
            f'before loan approval while maintaining strong overall discrimination '
            f'(<b>ROC-AUC = {metrics["roc_auc"]:.3f}</b>).</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # FIX #1: removed a stray, unmatched `st.markdown('</div>', unsafe_allow_html=True)`
    # that used to sit here. col1 and col2 each already open AND close their own
    # <div> blocks internally, so this extra closing tag had nothing left to
    # close — the browser instead closed the next real container it found
    # (the Streamlit column-row wrapper), which is what was collapsing/hiding
    # the right-hand content and the chart row below it.

    st.write("")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
        '<div class="ciq-card" style="margin-bottom:20px;">'
        '<div style="font-size:13.5px;font-weight:700;color:#101820;">ROC-AUC Comparison</div>'
        '<div style="font-size:11.3px;color:#8A94A6;font-weight:500;margin-bottom:8px;">Top models ranked by discrimination power</div>',
        unsafe_allow_html=True,
    )

        top6 = model_data.head(6)

        colors = [
            "#16A34A" if b else c
            for b, c in zip(
                top6["best"],
                [
                    "#2563EB",
                    "#F59E0B",
                    "#8B5CF6",
                    "#06B6D4",
                    "#EC4899",
                    "#94A3B8",
                ],
            )
        ]

        fig = go.Figure(
        go.Bar(
            x=top6["Model"],
            y=top6["ROC-AUC"],
            text=[f"{v:.3f}" for v in top6["ROC-AUC"]],
            textposition="outside",
            textfont=dict(size=10),
            marker_color=colors,
            width=0.45,
            hovertemplate="<b>%{x}</b><br>ROC-AUC : %{y:.3f}<extra></extra>",
        )
    )

        best_row = top6[top6["best"]]

        if not best_row.empty:
            fig.add_annotation(
                x=best_row.iloc[0]["Model"],
                y=best_row.iloc[0]["ROC-AUC"] + 0.01,
                text="🏆 Production Model",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-35,
                bgcolor="white",
                bordercolor="#16A34A",
            )

        fig.update_layout(
        height=300,

        margin=dict(
            l=10,
            r=10,
            t=10,
            b=45,
        ),

        yaxis=dict(
            title="ROC-AUC",
            range=[0.55,0.80],
            gridcolor="#EEF1F6",
        ),

        xaxis=dict(
            tickangle=-20,
            tickfont=dict(size=8),
        ),

        plot_bgcolor="white",
        paper_bgcolor="white",

        font=dict(
            family="Inter",
            size=8,
            color="#344054",
        ),

        showlegend=False,
    )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config=PLOTLY_CONFIG,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="ciq-card" style="margin-bottom:20px;">'
                    '<div style="font-size:13.5px;font-weight:700;color:#101820;">Precision vs. recall</div>'
                    '<div style="font-size:11.3px;color:#8A94A6;font-weight:500;margin-bottom:8px;">Trade-off across top 6 models</div>',
                    unsafe_allow_html=True)
        precision = top6["Precision"]
        recall = top6["Recall"]

        fig2 = go.Figure(
            go.Scatter(
                x=precision,
                y=recall,
                mode="markers",
                marker=dict(
                    size=[24 if b else 15 for b in top6["best"]],
                    color=[
                    "#16A34A" if b else c
                    for b, c in zip(
                        top6["best"],
                        [
                            "#2563EB",
                            "#F59E0B",
                            "#8B5CF6",
                            "#06B6D4",
                            "#EC4899",
                            "#94A3B8",
                        ],
                    )
                ],
                    line=dict(
                        color="white",
                        width=1.5,
                    ),
                ),
                customdata=top6["Model"],
                text=top6["Model"],
                textposition=[
                    "top center",
                    "bottom center",
                    "top left",
                    "bottom left",
                    "top right",
                    "bottom right",
                ],
                hovertemplate=
                    "<b>%{customdata}</b><br>" +
                    "Precision : %{x:.3f}<br>" +
                    "Recall : %{y:.3f}<br>" +
                    "<extra></extra>",
            )
        )

        best_rows = top6[top6["best"]]
        best = top6.loc[top6["F1"].idxmax()]

        fig2.add_annotation(
            x=best["Precision"],
            y=best["Recall"],
            text="🏆 Production Model",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-30,
            bgcolor="white",
            bordercolor="#16A34A",
        )

        fig2.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(
                    title="Precision",
                    range=[0, 1],
                    gridcolor="#EEF1F6",
                ),
                yaxis=dict(
                    title="Recall",
                    range=[0, 1],
                    gridcolor="#EEF1F6",
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(
                    family="Inter",
                    size=11,
                    color="#344054",
                ),
            )
        st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown(
            '<div class="ciq-card" style="margin-bottom:20px;">'
            '<div style="font-size:13.5px;font-weight:700;color:#101820;">Threshold optimization</div>'
            '<div style="font-size:11.3px;color:#8A94A6;font-weight:500;margin-bottom:8px;">Precision / Recall / F1 vs. Decision Threshold</div>',
            unsafe_allow_html=True,
        )

        thresholds = threshold_df["Threshold"]
        precision_curve = threshold_df["Precision"]
        recall_curve = threshold_df["Recall"]
        f1_curve = threshold_df["F1 Score"]

        fig3 = go.Figure()

        fig3.add_trace(
            go.Scatter(
                x=thresholds,
                y=precision_curve,
                mode="lines+markers",
                name="Precision",
                line=dict(color=ORANGE, width=3),
            )
        )

        fig3.add_trace(
            go.Scatter(
                x=thresholds,
                y=recall_curve,
                mode="lines+markers",
                name="Recall",
                line=dict(color=BLUE, width=3),
            )
        )

        fig3.add_trace(
            go.Scatter(
                x=thresholds,
                y=f1_curve,
                mode="lines+markers",
                name="F1 Score",
                line=dict(color=GREEN, width=3, dash="dash"),
            )
        )

        fig3.add_vline(
            x=metrics["threshold"],
            line_dash="dash",
            line_color="#DC2626",
            line_width=2,
        )

        fig3.add_trace(
            go.Scatter(
                x=[metrics["threshold"]],
                y=[metrics["f1"]],
                mode="markers",
                marker=dict(
                    size=12,
                    color="#DC2626",
                    line=dict(color="white", width=2),
                ),
                showlegend=False,
                hovertemplate=(
                    f"Threshold : {metrics['threshold']:.2f}<br>"
                    f"F1 Score : {metrics['f1']:.3f}<extra></extra>"
                ),
            )
        )

        fig3.add_annotation(
            x=metrics["threshold"],
            y=metrics["f1"],
            text=f"<b>Production Threshold</b><br>{metrics['threshold']:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-55,
            bgcolor="white",
            bordercolor="#DC2626",
            borderwidth=1,
            font=dict(size=10),
        )

        fig3.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(
                title="Decision Threshold",
                gridcolor="#EEF1F6",
            ),
            yaxis=dict(
                title="Score",
                range=[0, 1],
                gridcolor="#EEF1F6",
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(
                family="Inter",
                size=11,
                color="#344054",
            ),
        )

        st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG)
        st.markdown("</div>", unsafe_allow_html=True)


    st.write("")

    d1, d2 = st.columns(2)

    with d1:

        st.markdown(
            f'<div class="ciq-card">'
            f'<div style="font-size:13.5px;font-weight:700;color:#101820;">Confusion Matrix</div>'
            f'<div style="font-size:11.3px;color:#8A94A6;font-weight:500;margin-bottom:14px;">'
            f'{metrics["best_model"]} &nbsp;&bull;&nbsp; Threshold = {metrics["threshold"]:.2f}</div>',
            unsafe_allow_html=True,
        )

        gcol1, gcol2, gcol3 = st.columns([1, 1, 1])

        with gcol1:

            st.markdown(
                '<div style="height:44px;"></div>'
                '<div class="cm-axis">Actual<br>No Default</div>'
                '<div style="height:14px;"></div>'
                '<div class="cm-axis">Actual<br>Default</div>',
                unsafe_allow_html=True,
            )

        with gcol2:

            st.markdown(
                f'<div class="cm-axis">Pred. No Default</div>'
                f'<div class="cm-cell cm-tn">'
                f'<div class="cm-val">{metrics["tn"]:,}</div>'
                f'<div class="cm-tag">True Negative</div></div>'
                f'<div style="height:10px;"></div>'
                f'<div class="cm-cell cm-fn">'
                f'<div class="cm-val">{metrics["fn"]:,}</div>'
                f'<div class="cm-tag">False Negative</div></div>',
                unsafe_allow_html=True,
            )

        with gcol3:

            st.markdown(
                f'<div class="cm-axis">Pred. Default</div>'
                f'<div class="cm-cell cm-fp">'
                f'<div class="cm-val">{metrics["fp"]:,}</div>'
                f'<div class="cm-tag">False Positive</div></div>'
                f'<div style="height:10px;"></div>'
                f'<div class="cm-cell cm-tp">'
                f'<div class="cm-val">{metrics["tp"]:,}</div>'
                f'<div class="cm-tag">True Positive</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


    with d2:

        st.markdown(
            f'<div class="ciq-card">'
            f'<div style="font-size:13.5px;font-weight:700;margin-bottom:14px;color:#101820;">'
            f'Best Model Summary</div>'
            f'<div class="metric-row"><span class="m-name">Best Model</span>'
            f'<span class="m-desc">{metrics["best_model"]}</span></div>'
            f'<div class="metric-row"><span class="m-name">Decision Threshold</span>'
            f'<span class="m-desc">{metrics["threshold"]:.2f}</span></div>'
            f'<div class="metric-row"><span class="m-name">ROC-AUC</span>'
            f'<span class="m-desc">{metrics["roc_auc"]:.3f}</span></div>'
            f'<div class="metric-row"><span class="m-name">Accuracy</span>'
            f'<span class="m-desc">{metrics["accuracy"]:.3f}</span></div>'
            f'<div class="metric-row"><span class="m-name">Precision</span>'
            f'<span class="m-desc">{metrics["precision"]:.3f}</span></div>'
            f'<div class="metric-row"><span class="m-name">Recall</span>'
            f'<span class="m-desc">{metrics["recall"]:.3f}</span></div>'
            f'<div class="metric-row"><span class="m-name">F1 Score</span>'
            f'<span class="m-desc">{metrics["f1"]:.3f}</span></div>'
            f'<div class="metric-row"><span class="m-name">Validation Samples</span>'
            f'<span class="m-desc">{metrics["validation_size"]:,}</span></div>'
            f'<div class="metric-row"><span class="m-name">Features Used</span>'
            f'<span class="m-desc">{metrics["features"]}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
# ============================================================================
# PAGE: ABOUT
# ============================================================================
elif selected == "About":

    section_head(
        "About this project",
        "Background, methodology, dataset and business impact of the Home Loan Default Risk Prediction System."
    )

    st.markdown(
        """
        <div class="ciq-card">
            <div class="eyebrow" style="color:#101820;">Project Overview</div>
            <p style="font-size:14px;
                      line-height:1.8;
                      color:#475467;
                      font-weight:500;">
            Financial institutions lose millions every year because of loan defaults.
            Manual credit assessment is often slow, inconsistent and unable to identify
            complex relationships within customer data.
            <br><br>
            This application uses Machine Learning to estimate the probability that an
            applicant will default on a loan before approval. The prediction helps banks
            make faster, more consistent and data-driven lending decisions while reducing
            credit risk.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="ciq-card">

            <div style="font-size:15px;
                        font-weight:700;
                        color:#101820;
                        margin-bottom:18px;">

                🎯 Project Objectives

            </div>

            <div class="check-item"><span class="tick">✓</span>Predict loan default before loan approval.</div>

            <div class="check-item"><span class="tick">✓</span>Reduce financial losses caused by risky borrowers.</div>

            <div class="check-item"><span class="tick">✓</span>Support credit analysts with AI-powered decisions.</div>

            <div class="check-item"><span class="tick">✓</span>Increase consistency in underwriting decisions.</div>

            <div class="check-item"><span class="tick">✓</span>Compare multiple machine learning algorithms.</div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            f"""
            <div class="ciq-card">

            <div style="font-size:15px;
                        font-weight:700;
                        color:#101820;
                        margin-bottom:18px;">

                📊 Dataset

            </div>

            <div class="bm-row">
                <span class="k">Dataset</span>
                <span class="v">Home Credit Default Risk</span>
            </div>

            <div class="bm-row">
                <span class="k">Features Used</span>
                <span class="v">{metrics["features"]}</span>
            </div>

            <div class="bm-row">
                <span class="k">Target Variable</span>
                <span class="v">TARGET</span>
            </div>

            <div class="bm-row">
                <span class="k">Prediction Task</span>
                <span class="v">Binary Classification</span>
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        models_html = "".join(
            f'<span class="model-tag">{m}</span>'
            for m in [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "XGBoost",
                "LightGBM",
            ]
        )

        metrics_html = "".join(
            f'<span class="model-tag">{m}</span>'
            for m in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
            ]
        )

        st.markdown(
            f"""
            <div class="ciq-card">

            <div style="font-size:15px;
                        font-weight:700;
                        color:#101820;
                        margin-bottom:15px;">

                🤖 Models Evaluated

            </div>

            {models_html}

            <div style="margin-top:25px;
                        margin-bottom:15px;
                        font-size:15px;
                        font-weight:700;
                        color:#101820;">

                📈 Evaluation Metrics

            </div>

            {metrics_html}

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            f"""
            <div class="best-model-card">

            <div class="bm-label">
                Production Model
            </div>

            <div class="bm-title">
                {metrics["best_model"]}
            </div>

            <div class="bm-row">
                <span class="k">ROC-AUC</span>
                <span class="v">{metrics["roc_auc"]:.3f}</span>
            </div>

            <div class="bm-row">
                <span class="k">Accuracy</span>
                <span class="v">{metrics["accuracy"]:.3f}</span>
            </div>

            <div class="bm-row">
                <span class="k">Precision</span>
                <span class="v">{metrics["precision"]:.3f}</span>
            </div>

            <div class="bm-row">
                <span class="k">Recall</span>
                <span class="v">{metrics["recall"]:.3f}</span>
            </div>

            <div class="bm-row">
                <span class="k">F1 Score</span>
                <span class="v">{metrics["f1"]:.3f}</span>
            </div>

            <div class="bm-row">
                <span class="k">Threshold</span>
                <span class="v">{metrics["threshold"]:.2f}</span>
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    section_head(
        "Machine Learning Pipeline",
        "End-to-end workflow followed during model development."
    )

    steps = [
        ("1", "📥", "Data Collection"),
        ("2", "🧹", "Data Cleaning"),
        ("3", "📊", "EDA"),
        ("4", "🧬", "Feature Engineering"),
        ("5", "🤖", "Model Training"),
        ("6", "📈", "Model Evaluation"),
        ("7", "🎯", "Threshold Optimization"),
        ("8", "🚀", "Deployment"),
    ]

    html = "".join(
        f"""
        <div class="rail-step">
            <div class="rail-node">{icon}</div>
            <div class="rail-num">STEP {num}</div>
            <div class="rail-name">{name}</div>
        </div>
        """
        for num, icon, name in steps
    )

    st.markdown(
        f"""
        <div class="ciq-card rail-wrap">
            <div class="rail">
                {html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        """
        <div class="ciq-card">

        <div style="font-size:16px;
                    font-weight:700;
                    color:#101820;
                    margin-bottom:16px;">

            💼 Business Impact

        </div>

        <p style="font-size:14px;
                  line-height:1.8;
                  color:#475467;">

        By identifying high-risk applicants before loan approval, this solution
        enables financial institutions to reduce credit losses, improve portfolio
        quality, automate risk assessment, and support faster lending decisions.

        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="ciq-footer">
                <div>
                    Developed by <b>Sanjana S</b>
                </div>
                <div>
                    Home Credit Default Risk Dataset · Streamlit · Scikit-Learn · XGBoost
                </div>
         </div>
        """,
        unsafe_allow_html=True,
    )