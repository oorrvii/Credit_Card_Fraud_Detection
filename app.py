import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="FraudShield", page_icon="🛡️", layout="centered")

SAMPLES = {
    "legit_1": {"label": "Online shopping", "amount": 149.62, "emoji": "🛍️", "sub": "$149.62", "features": [-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215]},
    "legit_2": {"label": "ATM withdrawal", "amount": 2.69, "emoji": "🏧", "sub": "$2.69", "features": [1.19185711131486,0.26615071205963,0.16648011335321,0.448154078460911,0.0600176492822243,-0.0823608088155687,-0.0788029833323113,0.0851016549148104,-0.255425128109186,-0.166974414004614,1.61272666105479,1.06523531137287,0.48909501589608,-0.143772296441519,0.635558093258208,0.463917041022171,-0.114804663102346,-0.183361270123994,-0.145783041325259,-0.0690831352230203,-0.225775248033138,-0.638671952771851,0.101288021253234,-0.339846475529127,0.167170404418143,0.125894532368176,-0.0089830991432281,0.0147241691924927]},
    "legit_3": {"label": "Restaurant bill", "amount": 378.66, "emoji": "🍽️", "sub": "$378.66", "features": [-1.35835406159823,-1.34016307473609,1.77320934263119,0.379779593034328,-0.503198133318193,1.80049938079263,0.791460956450422,0.247675786588991,-1.51465432260583,0.207642865216696,0.624501459424895,0.066083685268831,0.717292731410831,-0.165945922763554,2.34586494901581,-2.89008319444231,1.10996937869599,-0.121359313195888,-2.26185709530414,0.524979725224404,0.247998153469754,0.771679401917229,0.909412262347719,-0.689280956490685,-0.327641833735251,-0.139096571514147,-0.0553527940384261,-0.0597518405929204]},
    "fraud_1": {"label": "Suspicious transfer", "amount": 0.0, "emoji": "⚡", "sub": "$0.00", "features": [-2.3122265423263,1.95199201064158,-1.60985073229769,3.9979055875468,-0.522187864667764,-1.42654531920595,-2.53738730624579,1.39165724829804,-2.77008927719433,-2.77227214465915,3.20203320709635,-2.89990738849473,-0.595221881324605,-4.28925378244217,0.389724120274487,-1.14074717980657,-2.83005567450437,-0.0168224681808257,0.416955705037907,0.126910559061474,0.517232370861764,-0.0350493686052974,-0.465211076182388,0.320198198514526,0.0445191674731724,0.177839798284401,0.261145002567677,-0.143275874698919]},
    "fraud_2": {"label": "Card skimming", "amount": 59.0, "emoji": "🌙", "sub": "$59.00", "features": [-4.39797444171999,1.35836702839758,-2.5928442182573,2.67978696694832,-1.12813094208956,-1.70653638774951,-3.49619729302467,-0.248777743025673,-0.24776789948008,-4.80163740602813,4.89584422347523,-10.9128193194019,0.184371685834387,-6.77109672468083,-0.0073261825777121,-7.35808322132346,-12.5984185405511,-5.13154862842983,0.308333945758691,-0.17160787864796,0.573574068424352,0.176967718048195,-0.436206883597401,-0.0535018648884285,0.252405261951833,-0.657487754764504,-0.827135714578603,0.849573379985768]},
    "fraud_3": {"label": "Duplicate card use", "amount": 1.0, "emoji": "💳", "sub": "$1.00", "features": [1.23423504613468,3.0197404207034,-4.30459688479665,4.73279513041887,3.62420083055386,-1.35774566315358,1.71344498787235,-0.496358487073991,-1.28285782036322,-2.44746925511151,2.10134386504854,-4.6096283906446,1.46437762476188,-6.07933719308005,-0.339237372732577,2.58185095378146,6.73938438478335,3.04249317830411,-2.72185312222835,0.0090608363953452,-0.37906830709218,-0.704181032215427,-0.656804756348389,-1.63265295692929,1.48890144838237,0.566797273468934,-0.0100162234965625,0.146792734916988]},
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

if "dark" not in st.session_state:
    st.session_state.dark = True

dark = st.session_state.dark

if dark:
    BG          = "#0b0e1a"
    CARD        = "#0e1120"
    BORDER      = "#1e2235"
    BORDER2     = "#252945"
    TEXT        = "#e2e8f0"
    MUTED       = "#4b5563"
    MUTED2      = "#374151"
    ACCENT      = "#6366f1"
    ACCENT_BG   = "#1e1f3b"
    ACCENT_BR   = "#3730a3"
    ACCENT_TX   = "#818cf8"
    STAT_BG     = "#0e1120"
    STAT_GRID   = "#161929"
    HOW_NUM     = "#1e2235"
    HOW_T       = "#9ca3af"
    S_BG        = "#0a1a0a"
    S_BR        = "#166534"
    S_TAG       = "#22c55e"
    S_TITLE     = "#4ade80"
    S_DESC      = "#374151"
    F_BG        = "#1a0a0a"
    F_BR        = "#991b1b"
    F_TAG       = "#ef4444"
    F_TITLE     = "#f87171"
    F_DESC      = "#374151"
    DIV         = "#161929"
    BTN_BG      = "#0e1120"
    BTN_BR      = "#1e2235"
    BTN_FRAUD   = "#2d1515"
    TOGGLE_BG   = "#161929"
    TOGGLE_BR   = "#252945"
    TOGGLE_TX   = "#6b7280"
    TOGGLE_ICON = "☀"
    TOGGLE_LBL  = "Light mode"
else:
    BG          = "#f8fafc"
    CARD        = "#ffffff"
    BORDER      = "#e2e8f0"
    BORDER2     = "#cbd5e1"
    TEXT        = "#0f172a"
    MUTED       = "#64748b"
    MUTED2      = "#94a3b8"
    ACCENT      = "#4f46e5"
    ACCENT_BG   = "#eef2ff"
    ACCENT_BR   = "#c7d2fe"
    ACCENT_TX   = "#4338ca"
    STAT_BG     = "#ffffff"
    STAT_GRID   = "#e2e8f0"
    HOW_NUM     = "#e2e8f0"
    HOW_T       = "#475569"
    S_BG        = "#f0fdf4"
    S_BR        = "#86efac"
    S_TAG       = "#16a34a"
    S_TITLE     = "#15803d"
    S_DESC      = "#64748b"
    F_BG        = "#fff1f2"
    F_BR        = "#fca5a5"
    F_TAG       = "#dc2626"
    F_TITLE     = "#b91c1c"
    F_DESC      = "#64748b"
    DIV         = "#e2e8f0"
    BTN_BG      = "#ffffff"
    BTN_BR      = "#e2e8f0"
    BTN_FRAUD   = "#fff1f2"
    TOGGLE_BG   = "#f1f5f9"
    TOGGLE_BR   = "#e2e8f0"
    TOGGLE_TX   = "#64748b"
    TOGGLE_ICON = "🌙"
    TOGGLE_LBL  = "Dark mode"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: {BG} !important;
    color: {TEXT} !important;
}}

.stApp {{ background: {BG} !important; }}

.topbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    padding-top: 0.5rem;
}}

.logo {{
    font-size: 1.15rem;
    font-weight: 800;
    color: {TEXT};
    letter-spacing: -0.03em;
}}

.logo span {{ color: {ACCENT}; }}

.hero {{
    text-align: center;
    margin-bottom: 1.8rem;
}}

.pill {{
    display: inline-block;
    background: {ACCENT_BG};
    border: 1px solid {ACCENT_BR};
    color: {ACCENT_TX};
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.25rem 0.85rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}}

.h1 {{
    font-size: 2.6rem;
    font-weight: 800;
    color: {TEXT};
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}}

.h1 span {{ color: {ACCENT}; }}

.sub {{
    font-size: 0.86rem;
    color: {MUTED};
    max-width: 340px;
    margin: 0 auto;
    line-height: 1.7;
}}

.stats {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: {STAT_GRID};
    border: 1px solid {STAT_GRID};
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1.8rem;
}}

.stat {{
    background: {STAT_BG};
    padding: 1.1rem 1rem;
    text-align: center;
}}

.stat-n {{
    font-size: 1.65rem;
    font-weight: 800;
    color: {ACCENT};
    letter-spacing: -0.03em;
    line-height: 1;
}}

.stat-l {{
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {MUTED2};
    margin-top: 3px;
}}

.sec {{
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {MUTED2};
    margin: 1.5rem 0 0.65rem;
}}

.result-legit {{
    background: {S_BG};
    border: 1px solid {S_BR};
    border-radius: 12px;
    padding: 1.8rem;
    text-align: center;
    margin-top: 1rem;
}}

.result-fraud {{
    background: {F_BG};
    border: 1px solid {F_BR};
    border-radius: 12px;
    padding: 1.8rem;
    text-align: center;
    margin-top: 1rem;
}}

.r-icon {{ font-size: 2rem; margin-bottom: 0.4rem; display: block; }}

.r-tag {{
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}}

.r-tag-l {{ color: {S_TAG}; }}
.r-tag-f {{ color: {F_TAG}; }}

.r-title {{
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.35rem;
}}

.r-title-l {{ color: {S_TITLE}; }}
.r-title-f {{ color: {F_TITLE}; }}
.r-desc {{ font-size: 0.82rem; color: {S_DESC}; line-height: 1.6; }}

.howgrid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.65rem;
    margin-top: 0.65rem;
}}

.howcard {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1rem;
}}

.how-n {{
    font-size: 1.5rem;
    font-weight: 800;
    color: {HOW_NUM};
    letter-spacing: -0.04em;
    margin-bottom: 0.4rem;
}}

.how-t {{
    font-size: 0.78rem;
    font-weight: 700;
    color: {HOW_T};
    margin-bottom: 0.2rem;
}}

.how-b {{ font-size: 0.71rem; color: {MUTED2}; line-height: 1.55; }}

.disc {{
    font-size: 0.63rem;
    color: {BORDER};
    text-align: center;
    margin-top: 2rem;
    line-height: 1.8;
}}

hr {{ border-color: {DIV} !important; margin: 1.5rem 0 !important; }}

div[data-testid="column"] .stButton > button {{
    background: {BTN_BG} !important;
    color: {TEXT} !important;
    border: 1px solid {BTN_BR} !important;
    border-radius: 10px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    padding: 0.8rem 0.4rem !important;
    width: 100% !important;
    line-height: 1.4 !important;
    transition: border-color 0.15s, color 0.15s !important;
}}

div[data-testid="column"] .stButton > button:hover {{
    border-color: {ACCENT} !important;
    color: {ACCENT} !important;
}}

.stButton[data-fraud="true"] > button {{
    background: {BTN_FRAUD} !important;
    border-color: #2d1515 !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Topbar ────────────────────────────────────────────────────────────────────
t1, t2 = st.columns([5, 1])
with t1:
    st.markdown(f'<div class="logo">Fraud<span>Shield</span></div>', unsafe_allow_html=True)
with t2:
    if st.button(f"{TOGGLE_ICON} {TOGGLE_LBL}", key="toggle"):
        st.session_state.dark = not st.session_state.dark
        st.rerun()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="pill">ML · Random Forest · Real-time</div>
    <div class="h1">Is this transaction<br><span>legit?</span></div>
    <div class="sub">Pick a sample transaction — the model classifies it instantly.</div>
</div>
""", unsafe_allow_html=True)

# ── Stats ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stats">
    <div class="stat"><div class="stat-n">93%</div><div class="stat-l">Precision</div></div>
    <div class="stat"><div class="stat-n">0.86</div><div class="stat-l">F1 Score</div></div>
    <div class="stat"><div class="stat-n">284K</div><div class="stat-l">Transactions</div></div>
</div>
""", unsafe_allow_html=True)

# ── Buttons ───────────────────────────────────────────────────────────────────
selected = None

st.markdown('<div class="sec">✓ Legitimate transactions</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🛍️ Online shopping\n$149.62", use_container_width=True, key="l1"): selected = "legit_1"
with c2:
    if st.button("🏧 ATM withdrawal\n$2.69", use_container_width=True, key="l2"): selected = "legit_2"
with c3:
    if st.button("🍽️ Restaurant bill\n$378.66", use_container_width=True, key="l3"): selected = "legit_3"

st.markdown('<div class="sec">⚠ Suspicious transactions</div>', unsafe_allow_html=True)
c4, c5, c6 = st.columns(3)
with c4:
    if st.button("⚡ Suspicious transfer\n$0.00", use_container_width=True, key="f1"): selected = "fraud_1"
with c5:
    if st.button("🌙 Card skimming\n$59.00", use_container_width=True, key="f2"): selected = "fraud_2"
with c6:
    if st.button("💳 Duplicate card\n$1.00", use_container_width=True, key="f3"): selected = "fraud_3"

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
                <span class="r-icon">✅</span>
                <div class="r-tag r-tag-l">Transaction cleared</div>
                <div class="r-title r-title-l">Legitimate</div>
                <div class="r-desc">{txn['emoji']} {txn['label']} · {txn['sub']} — no fraudulent patterns detected.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-fraud">
                <span class="r-icon">🚨</span>
                <div class="r-tag r-tag-f">Fraud detected</div>
                <div class="r-title r-title-f">Fraudulent</div>
                <div class="r-desc">{txn['emoji']} {txn['label']} · {txn['sub']} — flagged as fraudulent. Immediate review recommended.</div>
            </div>""", unsafe_allow_html=True)

# ── How it works ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="sec">How it works</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="howgrid">
    <div class="howcard"><div class="how-n">01</div><div class="how-t">Training data</div><div class="how-b">284,807 real transactions from the ULB Credit Card Fraud Dataset on Kaggle.</div></div>
    <div class="howcard"><div class="how-n">02</div><div class="how-t">Model</div><div class="how-b">Random Forest with balanced class weights for the 0.17% fraud imbalance.</div></div>
    <div class="howcard"><div class="how-n">03</div><div class="how-t">Result</div><div class="how-b">93% precision — when fraud is flagged, it's almost always real.</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disc">
    FOR EDUCATIONAL & PORTFOLIO PURPOSES ONLY · NOT FOR PRODUCTION USE IN FINANCIAL SYSTEMS<br>
    Dataset: ULB Machine Learning Group · Credit Card Fraud Detection (Kaggle)
</div>
""", unsafe_allow_html=True)