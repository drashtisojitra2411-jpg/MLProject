import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib
import plotly.express as px
import time
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# PAGE CONFIG
st.set_page_config(
    page_title="Typing Skill Analyzer",
    page_icon="⌨️",
    layout="wide",
)

# ADVANCED UI CSS + ANIMATED SHAPES + SOUND

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Nunito:wght@300;400;600;700&display=swap" rel="stylesheet">

<style>

body {
    background: linear-gradient(135deg, #0d1020, #111829, #141a2e);
    font-family: 'Poppins', sans-serif;
    overflow-x: hidden;
}

/* Floating Shapes */
.shape1, .shape2, .shape3 {
    position: fixed;
    border-radius: 50%;
    filter: blur(60px);
    opacity: 0.4;
    animation: float 9s infinite ease-in-out alternate;
}
.shape1 { width: 250px; height: 250px; background:#4a73ff; top:5%; left:5%; }
.shape2 { width: 350px; height: 350px; background:#ff5ec4; bottom:10%; right:10%; animation-duration: 11s; }
.shape3 { width: 300px; height: 300px; background:#00f0ff; top:50%; right:30%; animation-duration: 13s; }

@keyframes float {
    0% { transform: translateY(0px) translateX(0px); }
    100% { transform: translateY(-40px) translateX(40px); }
}

/* Page Title */
h1 {
    color: #e9edff;
    text-align: center;
    font-weight: 700;
    margin-bottom: 5px;
    animation: fadeSlide 1.2s ease-out;
}
@keyframes fadeSlide {
    0% { opacity: 0; transform: translateY(-10px) scale(0.95); }
    100% { opacity: 1; transform: translateY(0) scale(1); }
}

p {
    color: #aab4d4;
    text-align: center;
}

/* Glass Card */
.big-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    padding: 32px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 12px 30px rgba(0,0,0,0.45);
}

/* Typing Box */
textarea {
    background: rgba(255,255,255,0.09) !important;
    backdrop-filter: blur(6px) !important;
    color: #eaf0ff !important;
    border-radius: 14px !important;
    border: 1px solid #637cff !important;
    padding: 14px !important;
    font-size: 18px !important;
}

/* Dropdown */
div[data-baseweb="select"] {
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
}

/* Button */
div.stButton > button {
    background: linear-gradient(135deg, #506bff, #2948ff);
    color: white;
    border-radius: 14px;
    padding: 12px 32px;
    font-size: 17px;
    font-weight: 600;
    border: none;
}
div.stButton > button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 18px rgba(80, 107, 255, 0.55);
}

/* Badge Glow */
.badge {
    font-size: 20px;
    padding: 10px 20px;
    color: white;
    border-radius: 14px;
    font-weight: 600;
    display: inline-block;
    margin-top: 15px;
    animation: glowPulse 2s infinite alternate;
}
@keyframes glowPulse {
    0% { box-shadow: 0 0 8px rgba(255,255,255,0.2); }
    100% { box-shadow: 0 0 24px rgba(255,255,255,0.6); }
}

.gold { background:#f0c94d; color:#5c4600; }
.silver { background:#d7d7d7; color:#303030; }
.bronze { background:#c68c4a; color:#2a1b00; }
.bad { background:#ff4d4d; }

/* Footer */
.footer {
    margin-top: 40px;
    text-align: center;
    color: #7f8cb3;
}

</style>

<!-- Floating Shapes -->
<div class="shape1"></div>
<div class="shape2"></div>
<div class="shape3"></div>

<!-- Typing Sound -->
<audio id="typeSound" src="https://actions.google.com/sounds/v1/foley/click.ogg"></audio>
<script>
document.addEventListener("keydown", function() {
    document.getElementById("typeSound").play();
});
</script>

""", unsafe_allow_html=True)

# SESSION STATE
if "paragraph" not in st.session_state:
    st.session_state.paragraph = None
if "typed" not in st.session_state:
    st.session_state.typed = ""
if "results" not in st.session_state:
    st.session_state.results = None
if "start_time" not in st.session_state:
    st.session_state.start_time = None

# LOAD OR TRAIN MODEL
model_path = Path("typing_model.pkl")

if model_path.exists():
    model = joblib.load(model_path)
else:
    df = pd.read_csv("results.csv")[["wpm","acc","consistency","testDuration"]].dropna()

    def label_skill(r):
        if r["wpm"] < 40 or r["acc"] < 70:
            return "Beginner"
        elif r["wpm"] < 70:
            return "Intermediate"
        else:
            return "Advanced"

    df["skill"] = df.apply(label_skill, axis=1)
    X = df[["wpm","acc","consistency","testDuration"]]
    y = df["skill"]

    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, model_path)

# PARAGRAPH GENERATOR
def generate_paragraph(level):
    easy = [
        "Typing helps improve focus and builds confidence with daily practice.",
        "Practice daily to improve your typing accuracy and overall confidence."
    ]
    medium = [
        "Typing regularly improves communication productivity and digital fluency.",
        "Consistent practice helps develop accuracy rhythm and cognitive speed."
    ]
    hard = [
        "Artificial intelligence enhances human capability through computational precision and automation.",
        "Typing with precision requires mental coordination and structured cognitive movement."
    ]
    return random.choice({"Easy": easy, "Medium": medium, "Hard": hard}[level])

# MAIN UI
st.markdown("<h1>Typing Skill Analyzer</h1>", unsafe_allow_html=True)
st.write("<p>Smart ML-powered typing evaluation with animation, sound & glass effects</p>", unsafe_allow_html=True)

st.markdown("<div class='big-card'>", unsafe_allow_html=True)

st.subheader("Start Typing Test")

difficulty = st.selectbox("Choose Difficulty Level:", ["Easy", "Medium", "Hard"], index=1)

# Reset paragraph if difficulty changes
if st.session_state.paragraph is None or st.session_state.get("difficulty") != difficulty:
    st.session_state.paragraph = generate_paragraph(difficulty)
    st.session_state.difficulty = difficulty
    st.session_state.results = None
    st.session_state.start_time = None
    st.session_state.typed = ""

paragraph = st.session_state.paragraph

st.info(paragraph)

typed_text = st.text_area("Type the paragraph here:", value=st.session_state.typed, height=200)
st.session_state.typed = typed_text

# ------------ Start Timer ------------
if typed_text and st.session_state.start_time is None:
    st.session_state.start_time = time.time()

# ANALYZE PERFORMANCE
if st.button("Analyze Performance"):

    if not typed_text.strip():
        st.warning("⚠ Please type something before submitting.")
    else:

        p_words = paragraph.split()
        t_words = typed_text.split()

        correct = sum(1 for a, b in zip(p_words, t_words) if a == b)
        accuracy = round((correct / len(p_words)) * 100, 2)
        completion = round((len(t_words) / len(p_words)) * 100, 2)
        errors = len(p_words) - correct

        final_score = round((accuracy * 0.6) + (completion * 0.4), 2)

        # -------- TIME TAKEN --------
        end = time.time()
        time_taken = round(end - st.session_state.start_time, 2)

        # -------- WPM --------
        words_typed = len(t_words)
        wpm = round((words_typed / time_taken) * 60, 2) if time_taken > 0 else 0

        # Badge
        if final_score >= 90:
            badge = "🟡 Gold Performer"
            badge_class = "gold"
            st.balloons()
        elif final_score >= 70:
            badge = "⚪ Silver Performer"
            badge_class = "silver"
        elif final_score >= 50:
            badge = "🟤 Bronze Performer"
            badge_class = "bronze"
        else:
            badge = "❌ Needs Improvement"
            badge_class = "bad"

        features = np.array([[final_score, accuracy, completion, len(t_words)]])
        try:
            skill = model.predict(features)[0]
        except:
            skill = "Intermediate"

        st.session_state.results = {
            "accuracy": accuracy,
            "completion": completion,
            "errors": errors,
            "final_score": final_score,
            "skill": skill,
            "badge": badge,
            "badge_class": badge_class,
            "time_taken": time_taken,
            "wpm": wpm
        }

# SHOW RESULTS
if st.session_state.results:

    r = st.session_state.results

    st.markdown("## 📊 Performance Summary")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{r['accuracy']}%")
    c2.metric("Completion", f"{r['completion']}%")
    c3.metric("Errors", r["errors"])
    c4.metric("Final Score", r["final_score"])
    c5.metric("Time Taken", f"{r['time_taken']} sec")
    c6.metric("Speed", f"{r['wpm']} WPM")

    st.success(f"Skill Level Prediction: {r['skill']}")

    st.markdown(
        f"<div class='badge {r['badge_class']}'>{r['badge']}</div>",
        unsafe_allow_html=True
    )

    df_plot = pd.DataFrame({
        "Metric": ["Accuracy", "Completion", "Errors"],
        "Value": [r['accuracy'], r['completion'], r['errors']]
    })
    fig = px.bar(df_plot, x="Metric", y="Value", color="Metric", height=330)
    st.plotly_chart(fig, use_container_width=True)

# FOOTER
st.markdown("""
<div class='footer'>
    Designed & Developed by <b>Drashti Sojitra|Bhoomi Chandekar</b> — ML Project - WPM Accuracy (2025)<br>
    Powered by Streamlit · Machine Learning · Python
</div>
""", unsafe_allow_html=True)
