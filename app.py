import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="FraudShield", page_icon="🛡️", layout="centered")

# ── Real samples from dataset ─────────────────────────────────────────────────
SAMPLES = {
    "legit_1": {"label": "Online Shopping", "amount": 149.62, "emoji": "🛍️", "sub": "$149.62",
        "features": [-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215]},
    "legit_2": {"label": "ATM Withdrawal", "amount": 2.69, "emoji": "🏧", "sub": "$2.69",
        "features": [1.19185711131486,0.26615071205963,0.16648011335321,0.448154078460911,0.0600176492822243,-0.0823608088155687,-0.0788029833323113,0.0851016549148104,-0.255425128109186,-0.166974414004614,1.61272666105479,1.06523531137287,0.48909501589608,-0.143772296441519,0.635558093258208,0.463917041022171,-0.114804663102346,-0.183361270123994,-0.145783041325259,-0.0690831352230203,-0.225775248033138,-0.638671952771851,0.101288021253234,-0.339846475529127,0.167170404418143,0.125894532368176,-0.0089830991432281,0.0147241691924927]},
    "legit_3": {"label": "Restaurant Bill", "amount": 378.66, "emoji": "🍽️", "sub": "$378.66",
        "features": [-1.35835406159823,-1.34016307473609,1.77320934263119,0.379779593034328,-0.503198133318193,1.80049938079263,0.791460956450422,0.247675786588991,-1.51465432260583,0.207642865216696,0.624501459424895,0.066083685268831,0.717292731410831,-0.165945922763554,2.34586494901581,-2.89008319444231,1.10996937869599,-0.121359313195888,-2.26185709530414,0.524979725224404,0.247998153469754,0.771679401917229,0.909412262347719,-0.689280956490685,-0.327641833735251,-0.139096571514147,-0.0553527940384261,-0.0597518405929204]},
    "fraud_1": {"label": "Suspicious Transfer", "amount": 0.0, "emoji": "⚡", "sub": "$0.00",
        "features": [-2.3122265423263,1.95199201064158,-1.60985073229769,3.9979055875468,-0.522187864667764,-1.42654531920595,-2.53738730624579,1.39165724829804,-2.77008927719433,-2.77227214465915,3.20203320709635,-2.89990738849473,-0.595221881324605,-4.28925378244217,0.389724120274487,-1.14074717980657,-2.83005567450437,-0.0168224681808257,0.416955705037907,0.126910559061474,0.517232370861764,-0.0350493686052974,-0.465211076182388,0.320198198514526,0.0445191674731724,0.177839798284401,0.261145002567677,-0.143275874698919]},
    "fraud_2": {"label": "Card Skimming", "amount": 59.0, "emoji": "🌙", "sub": "$59.00",
        "features": [-4.39797444171999,1.35836702839758,-2.5928442182573,2.67978696694832,-1.12813094208956,-1.70653638774951,-3.49619729302467,-0.248777743025673,-0.24776789948008,-4.80163740602813,4.89584422347523,-10.9128193194019,0.184371685834387,-6.77109672468083,-0.0073261825777121,-7.35808322132346,-12.5984185405511,-5.13154862842983,0.308333945758691,-0.17160787864796,0.573574068424352,0.176967718048195,-0.436206883597401,-0.0535018648884285,0.252405261951833,-0.657487754764504,-0.827135714578603,0.849573379985768]},
    "fraud_3": {"label": "Duplicate Card Use", "amount": 1.0, "emoji": "💳", "sub": "$1.00",
        "features": [1.23423504613468,3.0197404207034,-4.30459688479665,4.73279513041887,3.62420083055386,-1.35774566315358,1.71344498787235,-0.496358487073991,-1.28285782036322,-2.44746925511151,2.10134386504854,-4.6096283906446,1.46437762476188,-6.07933719308005,-0.339237372732577,2.58185095378146,6.73938438478335,3.04249317830411,-2.72185312222835,0.0090608363953452,-0.37906830709218,-0.704181032215427,-0.656804756348389,-1.63265295692929,1.48890144838237,0.566797273468934,-0.0100162234965625,0.146792734916988]},
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

# ── Theme toggle ──────────────────────────────────────────────────────────────
if "dark" not in st.session_state:
    st.session_state.dark = False

col_toggle = st.columns([6, 1])
with col_toggle[1]:
    if st.button("🌙" if not st.session_state.dark else "☀️"):
        st.session_state.dark = not st.session_state.dark
        st.rerun()

dark = st.session_state.dark

# ── Theme tokens ──────────────────────────────────────────────────────────────
if dark:
    BG         = "#0f1117"
    CARD       = "#1a1f2e"
    BORDER     = "#252d3d"
    TEXT       = "#e2e8f0"
    MUTED      = "#475569"
    ACCENT     = "#3b82f6"
    ACCENT_BG  = "#1e3a5f"
    METRIC_BG  = "#141928"
    BTN_BG     = "#1a1f2e"
    BTN_HVR    = "#252d3d"
    DIVIDER    = "#1e2535"
    HOW_NUM    = "#252d3d"
    SUCCESS_BG = "#052e16"
    SUCCESS_BR = "#166534"
    SUCCESS_TT = "#4ade80"
    SUCCESS_TX = "#86efac"
    DANGER_BG  = "#2d0a0a"
    DANGER_BR  = "#991b1b"
    DANGER_TT  = "#f87171"
    DANGER_TX  = "#fca5a5"
else:
    BG         = "#f8fafc"
    CARD       = "#ffffff"
    BORDER     = "#e2e8f0"
    TEXT       = "#0f172a"
    MUTED      = "#64748b"
    ACCENT     = "#1d4ed8"
    ACCENT_BG  = "#eff6ff"
    METRIC_BG  = "#ffffff"
    BTN_BG     = "#ffffff"
    BTN_HVR    = "#f1f5f9"
    DIVIDER    = "#e2e8f0"
    HOW_NUM    = "#e2e8f0"
    SUCCESS_BG = "#f0fdf4"
    SUCCESS_BR = "#86efac"
    SUCCESS_TT = "#15803d"
    SUCCESS_TX = "#166534"
    DANGER_BG  = "#fff1f2"
    DANGER_BR  = "#fca5a5"
    DANGER_TT  = "#b91c1c"
    DANGER_TX  = "#991b1b"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Manrope', sans-serif !important;
    background-color: {BG} !important;
    color: {TEXT} !important;
}}

.stApp {{ background: {BG} !important; }}

.header {{
    padding: 1rem 0 1rem;
    text-align: center;
}}

.badge {{
    display: inline-block;
    background: {ACCENT_BG};
    color: {ACCENT};
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.28rem 0.85rem;
    border-radius: 20px;
    margin-bottom: 1rem;
    border: 1px solid {ACCENT}33;
}}

.title {{
    font-size: 2.8rem;
    font-weight: 800;
    color: {TEXT};
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}}

.title span {{ color: {ACCENT}; }}

.subtitle {{
    font-size: 0.88rem;
    color: {MUTED};
    max-width: 380px;
    margin: 0 auto;
    line-height: 1.7;
}}

.metrics {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 14px;
    overflow: hidden;
    margin: 1.8rem 0;
}}

.metric {{
    background: {METRIC_BG};
    padding: 1.1rem 1rem;
    text-align: center;
}}

.metric-val {{
    font-size: 1.75rem;
    font-weight: 800;
    color: {TEXT};
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 0.25rem;
}}

.metric-val span {{ color: {ACCENT}; }}
.metric-lbl {{
    font-size: 0.63rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {MUTED};
}}

.section-title {{
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {MUTED};
    margin: 1.6rem 0 0.7rem;
}}

.result-legit {{
    background: {SUCCESS_BG};
    border: 1.5px solid {SUCCESS_BR};
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.2rem;
}}

.result-fraud {{
    background: {DANGER_BG};
    border: 1.5px solid {DANGER_BR};
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.2rem;
}}

.result-icon {{ font-size: 2.4rem; margin-bottom: 0.5rem; display: block; }}

.result-tag {{
    font-size: 0.63rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}}

.tag-legit {{ color: {SUCCESS_TT}; }}
.tag-fraud {{ color: {DANGER_TT}; }}

.result-title {{
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.4rem;
}}

.rtitle-legit {{ color: {SUCCESS_TT}; }}
.rtitle-fraud {{ color: {DANGER_TT}; }}
.result-desc {{ font-size: 0.84rem; color: {MUTED}; line-height: 1.6; }}

.howgrid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-top: 0.75rem;
}}

.howcard {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1.1rem 1rem;
}}

.hownum {{
    font-size: 1.6rem;
    font-weight: 800;
    color: {HOW_NUM};
    letter-spacing: -0.04em;
    margin-bottom: 0.4rem;
}}

.howtitle {{
    font-size: 0.8rem;
    font-weight: 700;
    color: {TEXT};
    margin-bottom: 0.25rem;
}}

.howbody {{ font-size: 0.74rem; color: {MUTED}; line-height: 1.55; }}

.disc {{
    font-size: 0.65rem;
    color: {BORDER};
    text-align: center;
    margin-top: 2.5rem;
    line-height: 1.8;
}}

div[data-testid="column"] .stButton > button {{
    background: {BTN_BG} !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    padding: 0.65rem 0.5rem !important;
    width: 100% !important;
    line-height: 1.4 !important;
    transition: all 0.15s ease !important;
}}

div[data-testid="column"] .stButton > button:hover {{
    border-color: {ACCENT} !important;
    color: {ACCENT} !important;
    transform: translateY(-1px) !important;
}}

hr {{ border-color: {DIVIDER} !important; }}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header">
    <div class="badge">ML-Powered · Random Forest · Real-time</div>
    <div class="title">Fraud<span>Shield</span></div>
    <div class="subtitle">Pick a sample transaction to see the model classify it as legitimate or fraudulent instantly.</div>
</div>
""", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metrics">
    <div class="metric"><div class="metric-val"><span>93</span>%</div><div class="metric-lbl">Precision</div></div>
    <div class="metric"><div class="metric-val"><span>0.86</span></div><div class="metric-lbl">F1 Score</div></div>
    <div class="metric"><div class="metric-val"><span>284</span>K</div><div class="metric-lbl">Transactions</div></div>
</div>
""", unsafe_allow_html=True)

# ── Transaction buttons ───────────────────────────────────────────────────────
selected = None

st.markdown('<div class="section-title">✓ Legitimate Transactions</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🛍️ Online Shopping\n$149.62", use_container_width=True): selected = "legit_1"
with c2:
    if st.button("🏧 ATM Withdrawal\n$2.69", use_container_width=True): selected = "legit_2"
with c3:
    if st.button("🍽️ Restaurant Bill\n$378.66", use_container_width=True): selected = "legit_3"

st.markdown('<div class="section-title">⚠ Suspicious Transactions</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    if st.button("⚡ Suspicious Transfer\n$0.00", use_container_width=True): selected = "fraud_1"
with c5:
    if st.button("🌙 Card Skimming\n$59.00", use_container_width=True): selected = "fraud_2"
with c6:
    if st.button("💳 Duplicate Card\n$1.00", use_container_width=True): selected = "fraud_3"

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
            <div class="result-legit">
                <span class="result-icon">✅</span>
                <div class="result-tag tag-legit">Transaction Cleared</div>
                <div class="result-title rtitle-legit">Legitimate</div>
                <div class="result-desc"><strong>{txn['emoji']} {txn['label']} · {txn['sub']}</strong> — No fraudulent patterns detected. Transaction is safe to process.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-fraud">
                <span class="result-icon">🚨</span>
                <div class="result-tag tag-fraud">Fraud Detected</div>
                <div class="result-title rtitle-fraud">Fraudulent</div>
                <div class="result-desc"><strong>{txn['emoji']} {txn['label']} · {txn['sub']}</strong> — Flagged as a fraudulent transaction. Immediate review recommended.</div>
            </div>
            """, unsafe_allow_html=True)

# ── How it works ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="howgrid">
    <div class="howcard"><div class="hownum">01</div><div class="howtitle">Training data</div><div class="howbody">Trained on 284,807 real transactions from the ULB Credit Card Fraud Dataset on Kaggle.</div></div>
    <div class="howcard"><div class="hownum">02</div><div class="howtitle">Model</div><div class="howbody">Random Forest with balanced class weights to handle the extreme 0.17% fraud imbalance.</div></div>
    <div class="howcard"><div class="hownum">03</div><div class="howtitle">Result</div><div class="howbody">93% precision — when the model flags fraud, it's almost always a real fraudulent transaction.</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disc">
    FOR EDUCATIONAL & PORTFOLIO PURPOSES ONLY · NOT FOR PRODUCTION USE IN FINANCIAL SYSTEMS<br>
    Dataset: ULB Machine Learning Group · Credit Card Fraud Detection (Kaggle)
</div>
""", unsafe_allow_html=True)