"""
app.py — Period AI using Sentence Embeddings
No training needed! Just run: python3 -m streamlit run app.py
"""

import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from data import INTENT_EXAMPLES, ANSWERS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Period AI", page_icon="🌸", layout="centered")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

:root {
    --rose:  #c2185b;
    --blush: #f8bbd0;
    --deep:  #880e4f;
    --cream: #fff8f9;
    --text:  #3b1a2a;
    --muted: #9e6b80;
}

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #fff0f5 0%, #fff8f9 60%, #fce4ec 100%);
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

.header-wrap { text-align: center; padding: 2.5rem 0 1.2rem; }
.header-wrap h1 { font-family: 'DM Serif Display', serif; font-size: 3rem; color: var(--rose); margin: 0; }
.header-wrap p { color: var(--muted); font-size: 0.95rem; font-weight: 300; letter-spacing: 0.04em; margin-top: 0.3rem; }

.bubble-wrap { margin: 0.6rem 0; }
.bubble-user {
    background: var(--rose); color: white;
    padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    margin-left: auto; max-width: 78%; width: fit-content;
    font-size: 0.95rem; line-height: 1.5;
    box-shadow: 0 2px 8px rgba(194,24,91,0.18);
}
.bubble-ai {
    background: white; color: var(--text);
    padding: 0.85rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    max-width: 80%; width: fit-content;
    font-size: 0.95rem; line-height: 1.6;
    border: 1px solid var(--blush);
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.label { font-size: 0.72rem; font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.2rem; color: var(--muted); }
.label-right { text-align: right; }

.section-title { font-family: 'DM Serif Display', serif; font-size: 1.6rem; color: var(--deep); margin: 2.5rem 0 1rem; border-bottom: 2px solid var(--blush); padding-bottom: 0.4rem; }
.myth-card { background: white; border: 1px solid var(--blush); border-left: 4px solid var(--rose); border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 1rem; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
.myth-label { font-size: 0.72rem; font-weight: 500; letter-spacing: 0.1em; text-transform: uppercase; color: var(--rose); margin-bottom: 0.3rem; }
.myth-card h4 { margin: 0 0 0.4rem; font-family: 'DM Serif Display', serif; font-size: 1.05rem; }
.myth-card p { margin: 0 0 0.5rem; font-size: 0.9rem; color: #5a3a4a; line-height: 1.55; }
.myth-card a { font-size: 0.82rem; color: var(--rose); text-decoration: none; font-weight: 500; }
.myth-card a:hover { text-decoration: underline; }

.stTextInput > div > div > input {
    border: 1.5px solid var(--blush) !important; border-radius: 12px !important;
    padding: 0.7rem 1rem !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important; background: white !important; color: var(--text) !important;
}
.stTextInput > div > div > input:focus { border-color: var(--rose) !important; box-shadow: 0 0 0 3px rgba(194,24,91,0.08) !important; }
.stButton > button {
    background: var(--rose) !important; color: white !important; border: none !important;
    border-radius: 12px !important; padding: 0.65rem 1.8rem !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important; font-size: 0.95rem !important;
}
.stButton > button:hover { background: var(--deep) !important; }
hr.divider { border: none; border-top: 1.5px solid var(--blush); margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Load model & build index (cached) ────────────────────────────────────────
@st.cache_resource
def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Build flat list of (example, intent) pairs
    examples = []
    intents  = []
    for intent, phrases in INTENT_EXAMPLES.items():
        for phrase in phrases:
            examples.append(phrase)
            intents.append(intent)
    embeddings = model.encode(examples, convert_to_tensor=True)
    return model, embeddings, intents

model, embeddings, intents = load_model()

# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

# ── Predict ───────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.35

def get_answer(text: str) -> str:
    text = preprocess(text)
    query_embedding = model.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()

    if best_score < CONFIDENCE_THRESHOLD:
        return ANSWERS["out_of_scope"]

    intent = intents[best_idx]
    return ANSWERS.get(intent, ANSWERS["out_of_scope"])

# ── Myths data ────────────────────────────────────────────────────────────────
MYTHS = [
    {
        "myth": "You shouldn't exercise during your period.",
        "fact": "Exercise can actually help relieve cramps and improve mood. Light to moderate movement is generally beneficial.",
        "source": "Mayo Clinic",
        "url": "https://www.mayoclinic.org/diseases-conditions/menstrual-cramps/diagnosis-treatment/drc-20374944",
    },
    {
        "myth": "A 'normal' period is exactly 28 days.",
        "fact": "Cycle lengths from 21 to 35 days are all considered normal. Every body is different.",
        "source": "NHS UK",
        "url": "https://www.nhs.uk/conditions/periods/",
    },
    {
        "myth": "You can't get pregnant during your period.",
        "fact": "While less likely, pregnancy during a period is possible — sperm can survive up to 5 days.",
        "source": "Planned Parenthood",
        "url": "https://www.plannedparenthood.org/blog/can-you-get-pregnant-if-you-have-sex-during-your-period",
    },
    {
        "myth": "PMS is just 'being emotional' — it's not real.",
        "fact": "PMS is a recognized medical condition caused by hormonal changes, affecting up to 75% of people who menstruate.",
        "source": "ACOG",
        "url": "https://www.acog.org/womens-health/faqs/premenstrual-syndrome",
    },
    {
        "myth": "Painful periods are always normal.",
        "fact": "Mild discomfort is common, but severe pain may indicate endometriosis or fibroids. See a doctor if pain disrupts your life.",
        "source": "Endometriosis Foundation",
        "url": "https://www.endofound.org/endometriosis",
    },
]

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
    <h1>🌸 Period AI</h1>
    <p>Confronting misconceptions · Menstrual health information you can trust</p>
</div>
""", unsafe_allow_html=True)

# ── Chat ──────────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="bubble-wrap">
        <div class="label">Period AI</div>
        <div class="bubble-ai">
            Hello! 👋 I'm here to help with any questions about periods and menstrual health —
            no question is too small or embarrassing. What's on your mind?
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="bubble-wrap">
            <div class="label label-right">You</div>
            <div style="display:flex;justify-content:flex-end;">
                <div class="bubble-user">{msg["content"]}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bubble-wrap">
            <div class="label">Period AI</div>
            <div class="bubble-ai">{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input("msg", placeholder="Type your question...",
                               label_visibility="collapsed", key="user_input")
with col2:
    send = st.button("Send")

if send and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input.strip()})
    reply = get_answer(user_input.strip())
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

# ── Myths ─────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Period Myths & Information</div>', unsafe_allow_html=True)

for item in MYTHS:
    st.markdown(f"""
    <div class="myth-card">
        <div class="myth-label">❌ Myth</div>
        <h4>{item['myth']}</h4>
        <p>✅ <strong>Fact:</strong> {item['fact']}</p>
        <a href="{item['url']}" target="_blank">📖 Source: {item['source']}</a>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;color:#9e6b80;font-size:0.8rem;margin-top:2rem;padding-bottom:2rem;">
    Period AI is for informational purposes only and does not replace professional medical advice.<br>
    Always consult a qualified healthcare provider for personal health concerns.
</div>""", unsafe_allow_html=True)
