import streamlit as st
import pickle
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🛡️",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background-color: #080c14;
    color: #c8d0e0;
}

/* hero */
.hero {
    padding: 2.5rem 0 1rem;
    text-align: center;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(0, 200, 120, 0.08);
    border: 1px solid rgba(0, 200, 120, 0.25);
    color: #00c878;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 4px;
    margin-bottom: 1.2rem;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin-bottom: 0.6rem;
}

.hero-title span {
    color: #00c878;
}

.hero-sub {
    color: #4a5568;
    font-size: 0.9rem;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.6;
}

/* stats */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin: 1.8rem 0;
}

.stat-card {
    background: #0d1320;
    border: 1px solid #1a2035;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}

.stat-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #00c878;
    line-height: 1;
    margin-bottom: 0.3rem;
}

.stat-lbl {
    font-size: 0.62rem;
    color: #2d3748;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* section */
.section-head {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #2d3748;
    margin: 1.8rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.section-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1a2035;
}

/* result boxes */
.result-fraud {
    background: linear-gradient(135deg, rgba(255, 60, 80, 0.12), rgba(255, 60, 80, 0.03));
    border: 1px solid rgba(255, 60, 80, 0.35);
    border-radius: 10px;
    padding: 1.8rem;
    text-align: center;
    margin-top: 1.2rem;
}

.result-legit {
    background: linear-gradient(135deg, rgba(0, 200, 120, 0.12), rgba(0, 200, 120, 0.03));
    border: 1px solid rgba(0, 200, 120, 0.35);
    border-radius: 10px;
    padding: 1.8rem;
    text-align: center;
    margin-top: 1.2rem;
}

.result-eyebrow {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.result-eyebrow-fraud { color: #ff3c50; }
.result-eyebrow-legit { color: #00c878; }

.result-heading {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}

.result-heading-fraud { color: #ff6070; }
.result-heading-legit { color: #00c878; }

.result-body {
    font-size: 0.83rem;
    color: #4a5568;
    line-height: 1.6;
}

.divider {
    border: none;
    border-top: 1px solid #1a2035;
    margin: 1.5rem 0;
}

.disclaimer {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #1e2535;
    text-align: center;
    margin-top: 2rem;
    line-height: 1.8;
    letter-spacing: 0.04em;
}
</style>
""", unsafe_allow_html=True)

# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('fraud_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artifacts()
    loaded = True
except:
    loaded = False

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🛡️ ML-Powered · Real-time Detection</div>
    <div class="hero-title">Credit Card<br><span>Fraud Detector</span></div>
    <div class="hero-sub">
        Enter transaction details to instantly classify whether a transaction is legitimate or fraudulent.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Stats ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-val">93%</div>
        <div class="stat-lbl">Precision</div>
    </div>
    <div class="stat-card">
        <div class="stat-val">79%</div>
        <div class="stat-lbl">Recall</div>
    </div>
    <div class="stat-card">
        <div class="stat-val">0.86</div>
        <div class="stat-lbl">F1 Score</div>
    </div>
    <div class="stat-card">
        <div class="stat-val">RF</div>
        <div class="stat-lbl">Algorithm</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-head">Transaction Features (V1 – V14)</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
v_vals = {}

with col1:
    for i in range(1, 8):
        v_vals[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")

with col2:
    for i in range(8, 15):
        v_vals[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")

st.markdown('<div class="section-head">Transaction Features (V15 – V28)</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    for i in range(15, 22):
        v_vals[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")

with col4:
    for i in range(22, 29):
        v_vals[f'V{i}'] = st.number_input(f"V{i}", value=0.0, format="%.4f", key=f"v{i}")

st.markdown('<div class="section-head">Transaction Amount</div>', unsafe_allow_html=True)
amount = st.number_input("Amount (₹ / $)", min_value=0.0, value=100.0, format="%.2f")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Analyze Transaction →", use_container_width=True, type="primary"):
    if not loaded:
        st.error("Model files not found. Make sure fraud_classifier.pkl and scaler.pkl are in the same folder.")
    else:
        features = np.array([[v_vals[f'V{i}'] for i in range(1, 29)] + [amount]])
        features[0, -1] = scaler.transform([[amount]])[0][0]

        result = model.predict(features)

        if result[0] == 1:
            st.markdown("""
            <div class="result-fraud">
                <div class="result-eyebrow result-eyebrow-fraud">⚠ Classification Result</div>
                <div class="result-heading result-heading-fraud">Fraudulent Transaction</div>
                <div class="result-body">
                    This transaction has been flagged as <strong>fraudulent</strong>. 
                    Immediate review and card block is recommended.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-legit">
                <div class="result-eyebrow result-eyebrow-legit">✓ Classification Result</div>
                <div class="result-heading result-heading-legit">Legitimate Transaction</div>
                <div class="result-body">
                    This transaction appears <strong>legitimate</strong>. 
                    No fraudulent activity detected.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚠ FOR EDUCATIONAL PURPOSES ONLY · NOT FOR PRODUCTION USE IN FINANCIAL SYSTEMS<br>
    Dataset: ULB Machine Learning Group · Credit Card Fraud Detection (Kaggle)
</div>
""", unsafe_allow_html=True)