import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="FraudShield", page_icon="🛡️", layout="centered")

SAMPLES = {
    "legit_1": {"label": "Online Shopping", "amount": 1240.00, "emoji": "🛍️", "sub": "₹1,240", "features": [-1.3598071,-0.0727812,2.5363467,1.3782155,-0.3383208,0.4623878,0.2395986,0.0986980,0.3637870,0.0907942,-0.5515995,-0.6178009,-0.9913898,-0.3111694,1.4681770,-0.4704005,0.2079712,0.0257906,0.4030606,0.2514120,-0.0183068,0.2778376,-0.1104838,0.0669281,0.1285394,-0.1891484,0.1335584,-0.0210531]},
    "legit_2": {"label": "ATM Withdrawal", "amount": 500.00, "emoji": "🏧", "sub": "₹500", "features": [1.1918571,0.2661507,0.1664801,0.4481541,0.0600176,-0.0823608,-0.0788030,0.0851017,-0.2554251,-0.1669823,1.6127267,1.0651586,0.4898992,-0.1437723,0.6355581,0.4639170,-0.1148526,-0.1833366,-0.1457837,-0.0690831,-0.2251884,0.1783516,0.5077569,-0.2879237,0.0399759,-0.0197154,0.0603734,-0.1500800]},
    "legit_3": {"label": "Restaurant Bill", "amount": 3800.00, "emoji": "🍽️", "sub": "₹3,800", "features": [-0.9932736,0.6716741,0.8085881,0.1477490,-0.7297212,-0.3552570,0.1652626,-0.1278382,0.1353592,-0.1377059,-0.4577221,-0.2830584,0.2456729,-0.3614764,0.2121751,0.2131074,0.1415211,-0.1046553,-0.0714060,0.0555940,0.0614166,-0.0615372,0.0679671,-0.0214592,0.0297429,-0.0363990,0.0180246,-0.0247251]},
    "fraud_1": {"label": "Suspicious Transfer", "amount": 18990.00, "emoji": "⚡", "sub": "₹18,990", "features": [-2.3122265,1.9519673,-1.6098041,3.9979055,-0.5220785,-1.4265408,-2.5373073,1.3916167,-2.7700382,-2.7722739,1.8066569,-0.9467338,-0.6213090,-1.0793534,-0.1682613,1.5121419,1.0631907,-0.3573956,0.5048835,-0.4564060,-0.1518032,-0.7505001,-0.6222521,-0.0752436,-0.2254895,-0.6382088,-0.0393525,-0.2327291]},
    "fraud_2": {"label": "Midnight Purchase", "amount": 9999.00, "emoji": "🌙", "sub": "₹9,999", "features": [-3.0435406,-3.1572426,1.0880174,2.2886436,1.3596491,-1.0933665,-1.2228892,-0.3580215,0.0765038,-0.5289668,-0.7161697,-0.5659023,0.3508930,0.9204526,-0.1974664,0.4880490,-0.1202048,0.7476621,0.4316903,-0.1720761,0.1058155,0.0530174,-0.0534453,0.0311989,-0.0476503,-0.2380388,0.0311578,-0.0256044]},
    "fraud_3": {"label": "Duplicate Card Use", "amount": 74500.00, "emoji": "💳", "sub": "₹74,500", "features": [1.9914716,0.7135843,-1.2084556,0.2046556,0.6245019,-1.1796996,-0.5704684,-1.7237317,0.1063937,-0.5016006,-0.6933773,0.4713960,0.0489048,-0.2401802,-1.3516614,0.5765946,-0.1997796,0.5028553,-0.4575419,0.3337451,-0.3126984,-0.0399621,0.0657549,-0.0393473,0.0228404,0.0484791,-0.1000181,0.0342699]},
}

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
except:
    loaded = False

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif !important;
}

.stApp { background: #f7f8fa; }

.header {
    padding: 2.5rem 0 1rem;
    text-align: center;
}

.badge {
    display: inline-block;
    background: #e8f0fe;
    color: #1a56db;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.28rem 0.75rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}

.title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}

.title span { color: #1a56db; }

.subtitle {
    font-size: 0.9rem;
    color: #64748b;
    max-width: 380px;
    margin: 0 auto;
    line-height: 1.7;
}

.metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: #e2e8f0;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    overflow: hidden;
    margin: 2rem 0;
}

.metric {
    background: #ffffff;
    padding: 1.2rem 1rem;
    text-align: center;
}

.metric-val {
    font-size: 1.75rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 0.25rem;
}

.metric-val span { color: #1a56db; }

.metric-lbl {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
}

.section-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 1.8rem 0 0.8rem;
}

.result-legit {
    background: #f0fdf4;
    border: 1.5px solid #86efac;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.2rem;
}

.result-fraud {
    background: #fff1f2;
    border: 1.5px solid #fca5a5;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.2rem;
}

.result-icon { font-size: 2.5rem; margin-bottom: 0.5rem; display: block; }

.result-tag {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

.tag-legit { color: #16a34a; }
.tag-fraud { color: #dc2626; }

.result-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.4rem;
}

.rtitle-legit { color: #15803d; }
.rtitle-fraud { color: #b91c1c; }

.result-desc { font-size: 0.85rem; color: #64748b; line-height: 1.6; }

.howgrid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.8rem;
    margin-top: 0.8rem;
}

.howcard {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.1rem 1rem;
}

.hownum {
    font-size: 1.6rem;
    font-weight: 800;
    color: #e2e8f0;
    letter-spacing: -0.04em;
    margin-bottom: 0.5rem;
}

.howtitle {
    font-size: 0.8rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.3rem;
}

.howbody { font-size: 0.75rem; color: #94a3b8; line-height: 1.55; }

.disc {
    font-size: 0.68rem;
    color: #cbd5e1;
    text-align: center;
    margin-top: 2.5rem;
    line-height: 1.8;
}

div[data-testid="column"] .stButton > button {
    background: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 0.7rem 0.5rem !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
    line-height: 1.4 !important;
}

div[data-testid="column"] .stButton > button:hover {
    border-color: #1a56db !important;
    color: #1a56db !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <div class="badge">ML-Powered · Random Forest</div>
    <div class="title">Fraud<span>Shield</span></div>
    <div class="subtitle">Pick a sample transaction to see the model classify it as legitimate or fraudulent in real time.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="metrics">
    <div class="metric"><div class="metric-val"><span>93</span>%</div><div class="metric-lbl">Precision</div></div>
    <div class="metric"><div class="metric-val"><span>0.86</span></div><div class="metric-lbl">F1 Score</div></div>
    <div class="metric"><div class="metric-val"><span>284</span>K</div><div class="metric-lbl">Transactions</div></div>
</div>
""", unsafe_allow_html=True)

selected = None

st.markdown('<div class="section-title">✓ Legitimate Transactions</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🛍️ Online Shopping\n₹1,240", use_container_width=True): selected = "legit_1"
with c2:
    if st.button("🏧 ATM Withdrawal\n₹500", use_container_width=True): selected = "legit_2"
with c3:
    if st.button("🍽️ Restaurant Bill\n₹3,800", use_container_width=True): selected = "legit_3"

st.markdown('<div class="section-title">⚠ Suspicious Transactions</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    if st.button("⚡ Suspicious Transfer\n₹18,990", use_container_width=True): selected = "fraud_1"
with c5:
    if st.button("🌙 Midnight Purchase\n₹9,999", use_container_width=True): selected = "fraud_2"
with c6:
    if st.button("💳 Duplicate Card\n₹74,500", use_container_width=True): selected = "fraud_3"

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
            <div class="result-legit">
                <span class="result-icon">✅</span>
                <div class="result-tag tag-legit">Transaction Cleared</div>
                <div class="result-title rtitle-legit">Legitimate</div>
                <div class="result-desc"><strong>{txn['emoji']} {txn['label']} · {txn['sub']}</strong> has been analyzed and classified as a legitimate transaction. No fraudulent patterns detected.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-fraud">
                <span class="result-icon">🚨</span>
                <div class="result-tag tag-fraud">Fraud Detected</div>
                <div class="result-title rtitle-fraud">Fraudulent</div>
                <div class="result-desc"><strong>{txn['emoji']} {txn['label']} · {txn['sub']}</strong> has been flagged as a fraudulent transaction. Immediate review recommended.</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
st.markdown("""
<div class="howgrid">
    <div class="howcard"><div class="hownum">01</div><div class="howtitle">Training Data</div><div class="howbody">Trained on 284,807 real transactions from the ULB Credit Card Fraud Dataset on Kaggle.</div></div>
    <div class="howcard"><div class="hownum">02</div><div class="howtitle">Model</div><div class="howbody">Random Forest with balanced class weights handles the extreme 0.17% fraud imbalance.</div></div>
    <div class="howcard"><div class="hownum">03</div><div class="howtitle">Result</div><div class="howbody">93% precision — when the model flags fraud, it's almost always a real fraudulent transaction.</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disc">
    FOR EDUCATIONAL & PORTFOLIO PURPOSES ONLY · NOT FOR PRODUCTION USE IN FINANCIAL SYSTEMS<br>
    Dataset: ULB Machine Learning Group · Credit Card Fraud Detection (Kaggle)
</div>
""", unsafe_allow_html=True)