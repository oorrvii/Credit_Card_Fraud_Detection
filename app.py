import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="FraudShield", page_icon="🛡️", layout="centered")

# ── Sample transactions (real-looking, anonymized) ────────────────────────────
SAMPLES = {
    "legit_1": {
        "label": "Online Shopping · ₹1,240",
        "amount": 1240.00,
        "tag": "E-commerce",
        "features": [-1.3598071,-0.0727812,2.5363467,1.3782155,-0.3383208,0.4623878,0.2395986,0.0986980,0.3637870,0.0907942,-0.5515995,-0.6178009,-0.9913898,-0.3111694,1.4681770,-0.4704005,0.2079712,0.0257906,0.4030606,0.2514120,-0.0183068,0.2778376,-0.1104838,0.0669281,0.1285394,-0.1891484,0.1335584,-0.0210531],
    },
    "legit_2": {
        "label": "ATM Withdrawal · ₹500",
        "amount": 500.00,
        "tag": "ATM",
        "features": [1.1918571,0.2661507,0.1664801,0.4481541,0.0600176,-0.0823608,-0.0788030,0.0851017,-0.2554251,-0.1669823,1.6127267,1.0651586,0.4898992,-0.1437723,0.6355581,0.4639170,-0.1148526,-0.1833366,-0.1457837,-0.0690831,-0.2251884,0.1783516,0.5077569,-0.2879237,0.0399759,-0.0197154,0.0603734,-0.1500800],
    },
    "legit_3": {
        "label": "Restaurant Bill · ₹3,800",
        "amount": 3800.00,
        "tag": "Dining",
        "features": [-0.9932736,0.6716741,0.8085881,0.1477490,-0.7297212,-0.3552570,0.1652626,-0.1278382,0.1353592,-0.1377059,-0.4577221,-0.2830584,0.2456729,-0.3614764,0.2121751,0.2131074,0.1415211,-0.1046553,-0.0714060,0.0555940,0.0614166,-0.0615372,0.0679671,-0.0214592,0.0297429,-0.0363990,0.0180246,-0.0247251],
    },
    "fraud_1": {
        "label": "Suspicious Transfer · ₹18,990",
        "amount": 18990.00,
        "tag": "Wire Transfer",
        "features": [-2.3122265,1.9519673,-1.6098041,3.9979055,-0.5220785,-1.4265408,-2.5373073,1.3916167,-2.7700382,-2.7722739,1.8066569,-0.9467338,-0.6213090,-1.0793534,-0.1682613,1.5121419,1.0631907,-0.3573956,0.5048835,-0.4564060,-0.1518032,-0.7505001,-0.6222521,-0.0752436,-0.2254895,-0.6382088,-0.0393525,-0.2327291],
    },
    "fraud_2": {
        "label": "Midnight Purchase · ₹9,999",
        "amount": 9999.00,
        "tag": "Flagged",
        "features": [-3.0435406,-3.1572426,1.0880174,2.2886436,1.3596491,-1.0933665,-1.2228892,-0.3580215,0.0765038,-0.5289668,-0.7161697,-0.5659023,0.3508930,0.9204526,-0.1974664,0.4880490,-0.1202048,0.7476621,0.4316903,-0.1720761,0.1058155,0.0530174,-0.0534453,0.0311989,-0.0476503,-0.2380388,0.0311578,-0.0256044],
    },
    "fraud_3": {
        "label": "Duplicate Card · ₹74,500",
        "amount": 74500.00,
        "tag": "High Risk",
        "features": [1.9914716,0.7135843,-1.2084556,0.2046556,0.6245019,-1.1796996,-0.5704684,-1.7237317,0.1063937,-0.5016006,-0.6933773,0.4713960,0.0489048,-0.2401802,-1.3516614,0.5765946,-0.1997796,0.5028553,-0.4575419,0.3337451,-0.3126984,-0.0399621,0.0657549,-0.0393473,0.0228404,0.0484791,-0.1000181,0.0342699],
    },
}

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('fraud_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_model()
    loaded = True
except Exception as e:
    loaded = False

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #06080f; color: #a0aab8; }

/* ── hero ── */
.hero { padding: 3rem 0 0.5rem; text-align: center; }

.shield-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    display: block;
    filter: drop-shadow(0 0 24px rgba(99,179,237,0.4));
}

.hero-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4a90d9;
    margin-bottom: 0.8rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.04em;
    line-height: 1.05;
    margin-bottom: 0.8rem;
}

.hero-title .blue { color: #4a90d9; }

.hero-desc {
    font-size: 0.88rem;
    color: #3d4a5c;
    max-width: 400px;
    margin: 0 auto 0.5rem;
    line-height: 1.7;
    font-weight: 400;
}

/* ── metrics row ── */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: #0f1520;
    border: 1px solid #0f1520;
    border-radius: 12px;
    overflow: hidden;
    margin: 2rem 0;
}

.metric-cell {
    background: #080c14;
    padding: 1.2rem 1rem;
    text-align: center;
}

.metric-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: #4a90d9;
    line-height: 1;
    margin-bottom: 0.3rem;
    letter-spacing: -0.03em;
}

.metric-label {
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #1e2d3d;
}

/* ── section header ── */
.sec-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #1e2d3d;
    margin: 2rem 0 1rem;
}

/* ── transaction cards ── */
.cards-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.6rem;
    margin-bottom: 0.5rem;
}

.txn-card {
    background: #080c14;
    border: 1px solid #0f1929;
    border-radius: 10px;
    padding: 0.9rem 0.8rem;
    cursor: pointer;
    transition: all 0.15s ease;
    text-align: left;
}

.txn-card:hover { border-color: #1e3a5f; background: #0a0f1a; }
.txn-card.selected { border-color: #4a90d9; background: rgba(74,144,217,0.06); }
.txn-card.fraud-card { border-color: #1a0f0f; }
.txn-card.fraud-card:hover { border-color: #5f1e1e; }
.txn-card.fraud-card.selected { border-color: #c0392b; background: rgba(192,57,43,0.06); }

.card-tag {
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.tag-legit { color: #2e86c1; }
.tag-fraud { color: #922b21; }

.card-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #6b7a8d;
    line-height: 1.3;
}

/* ── divider ── */
.div { border: none; border-top: 1px solid #0c1220; margin: 1.5rem 0; }

/* ── result ── */
.result-wrap { margin-top: 1.2rem; border-radius: 12px; padding: 2rem; text-align: center; }

.result-legit {
    background: linear-gradient(135deg, rgba(32,96,160,0.14), rgba(32,96,160,0.04));
    border: 1px solid rgba(74,144,217,0.3);
}

.result-fraud {
    background: linear-gradient(135deg, rgba(160,32,32,0.14), rgba(160,32,32,0.04));
    border: 1px solid rgba(192,57,43,0.3);
}

.result-icon { font-size: 2.2rem; margin-bottom: 0.6rem; display: block; }

.result-eyebrow {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.eyebrow-legit { color: #4a90d9; }
.eyebrow-fraud { color: #c0392b; }

.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.5rem;
}

.title-legit { color: #5dade2; }
.title-fraud { color: #e74c3c; }

.result-body { font-size: 0.83rem; color: #2e4057; line-height: 1.6; }

/* ── how it works ── */
.how-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin-top: 0.8rem;
}

.how-card {
    background: #080c14;
    border: 1px solid #0c1525;
    border-radius: 10px;
    padding: 1rem;
}

.how-num {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #0f1929;
    margin-bottom: 0.4rem;
}

.how-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #2e4057;
    margin-bottom: 0.3rem;
}

.how-body { font-size: 0.72rem; color: #1a2535; line-height: 1.5; }

/* ── disclaimer ── */
.disclaimer {
    font-size: 0.65rem;
    color: #0f1929;
    text-align: center;
    margin-top: 2.5rem;
    line-height: 1.8;
    letter-spacing: 0.04em;
}

/* button overrides */
.stButton > button {
    background: #4a90d9 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    background: #357abd !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="shield-icon">🛡️</span>
    <div class="hero-eyebrow">ML-Powered · Random Forest · Real-time</div>
    <div class="hero-title">Fraud<span class="blue">Shield</span></div>
    <div class="hero-desc">
        Select a sample transaction below to see the model classify it as legitimate or fraudulent in real time.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="metrics-row">
    <div class="metric-cell">
        <div class="metric-num">93%</div>
        <div class="metric-label">Precision</div>
    </div>
    <div class="metric-cell">
        <div class="metric-num">0.86</div>
        <div class="metric-label">F1 Score</div>
    </div>
    <div class="metric-cell">
        <div class="metric-num">284K</div>
        <div class="metric-label">Transactions Trained</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Legit transactions ────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">✓ Legitimate Transactions</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
selected = None

with col1:
    if st.button("🛍️ Online Shopping\n₹1,240", use_container_width=True):
        selected = "legit_1"
with col2:
    if st.button("🏧 ATM Withdrawal\n₹500", use_container_width=True):
        selected = "legit_2"
with col3:
    if st.button("🍽️ Restaurant Bill\n₹3,800", use_container_width=True):
        selected = "legit_3"

# ── Fraud transactions ────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">⚠ Suspicious Transactions</div>', unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    if st.button("⚡ Suspicious Transfer\n₹18,990", use_container_width=True):
        selected = "fraud_1"
with col5:
    if st.button("🌙 Midnight Purchase\n₹9,999", use_container_width=True):
        selected = "fraud_2"
with col6:
    if st.button("💀 Duplicate Card\n₹74,500", use_container_width=True):
        selected = "fraud_3"

# ── Result ────────────────────────────────────────────────────────────────────
if selected:
    if not loaded:
        st.error("Model files not found. Make sure fraud_classifier.pkl and scaler.pkl are present.")
    else:
        txn = SAMPLES[selected]
        features = np.array([txn["features"] + [txn["amount"]]])
        features_scaled = scaler.transform(features)
        result = model.predict(features_scaled)[0]

        if result == 0:
            st.markdown(f"""
            <div class="result-wrap result-legit">
                <span class="result-icon">✅</span>
                <div class="result-eyebrow eyebrow-legit">Transaction Cleared</div>
                <div class="result-title title-legit">Legitimate</div>
                <div class="result-body">
                    <strong>{txn['label']}</strong> has been analyzed and classified as a legitimate transaction.
                    No fraudulent patterns detected.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-wrap result-fraud">
                <span class="result-icon">🚨</span>
                <div class="result-eyebrow eyebrow-fraud">Alert: Fraud Detected</div>
                <div class="result-title title-fraud">Fraudulent</div>
                <div class="result-body">
                    <strong>{txn['label']}</strong> has been flagged as a fraudulent transaction.
                    Immediate review recommended.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── How it works ──────────────────────────────────────────────────────────────
st.markdown('<hr class="div">', unsafe_allow_html=True)
st.markdown('<div class="sec-header">How It Works</div>', unsafe_allow_html=True)
st.markdown("""
<div class="how-grid">
    <div class="how-card">
        <div class="how-num">01</div>
        <div class="how-title">Data</div>
        <div class="how-body">Trained on 284,807 real transactions from the ULB Credit Card Fraud Dataset.</div>
    </div>
    <div class="how-card">
        <div class="how-num">02</div>
        <div class="how-title">Model</div>
        <div class="how-body">Random Forest with balanced class weights handles the 0.17% fraud imbalance.</div>
    </div>
    <div class="how-card">
        <div class="how-num">03</div>
        <div class="how-title">Result</div>
        <div class="how-body">93% precision means when fraud is flagged, it's almost always real fraud.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    FOR EDUCATIONAL & PORTFOLIO PURPOSES ONLY · NOT FOR PRODUCTION USE<br>
    Dataset: ULB Machine Learning Group · Credit Card Fraud Detection (Kaggle)
</div>
""", unsafe_allow_html=True)