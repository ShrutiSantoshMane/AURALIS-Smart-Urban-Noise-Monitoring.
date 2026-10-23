from pathlib import Path
from datetime import datetime
import random
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ==========================================================
# AURALIS | Smart Urban Noise Monitoring System
# Reliable local Streamlit dashboard
# ==========================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PREDICTIONS_FILE = DATA_DIR / "akurdi_batch_predictions.csv"
HISTORY_FILE = DATA_DIR / "akurdi_real_event_history.csv"
HOTSPOTS_FILE = DATA_DIR / "akurdi_real_hotspot_priority.csv"
MAP_FILE = DATA_DIR / "akurdi_real_audio_hotspots_map.html"

st.set_page_config(
    page_title="Auralis | Urban Noise Intelligence",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple CSS only; no external fonts, JavaScript, or chart libraries.
st.markdown("""
<style>
.stApp { background: radial-gradient(circle at 10% 0%, #12344a 0%, #091522 42%, #060b12 100%); color: #edf7ff; }
[data-testid="stHeader"] { background: rgba(5,12,20,.95); }
[data-testid="stSidebar"] { background: #0b1927; border-right: 1px solid #254459; }
.block-container { max-width: 1400px; padding-top: 1.4rem; padding-bottom: 2.5rem; }
h1,h2,h3,h4 { color: #f3fbff !important; }
p, label, .stCaption { color: #b8cddd; }
.hero { padding: 28px 30px; border: 1px solid #2b596e; border-radius: 22px; background: linear-gradient(120deg,#12344a,#0b1b2c 65%,#10343a); margin-bottom: 18px; box-shadow: 0 14px 38px rgba(0,0,0,.18); }
.kicker { color: #62ead8; font-size: .72rem; letter-spacing: .18em; font-weight: 800; }
.hero-title { font-size: clamp(2rem,4vw,3.15rem); font-weight: 900; line-height: 1.06; margin: 10px 0; color: #f6fcff; }
.hero-title span { color: #62ead8; }
.hero-copy { max-width: 850px; line-height: 1.6; color: #bdd2df; }
.pill { display:inline-block; margin: 5px 5px 0 0; padding: 6px 10px; border-radius: 30px; border: 1px solid #315d70; background: #0a1c2b; color: #d8f2f4; font-size: .7rem; font-weight: 700; }
.note { margin: 12px 0 18px; padding: 12px 15px; background:#102535; border:1px solid #2c5266; border-radius:12px; color:#cee5ed; }
div[data-testid="stMetric"] { background:linear-gradient(145deg,#122c40,#0a1927); border:1px solid #2a4b60; border-radius:15px; padding:14px 16px; }
div[data-testid="stMetricLabel"] { color:#a8c0d0 !important; }
div[data-testid="stMetricValue"] { color:#f6fcff !important; }
div.stButton > button, div.stDownloadButton > button { border:1px solid #3d8e91; border-radius:10px; background:linear-gradient(120deg,#176c78,#16516b); color:#fff; font-weight:750; }
hr { border-color:#284658; }
</style>
""", unsafe_allow_html=True)


def read_csv_safe(path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.sidebar.warning(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


def normalize_columns(df, aliases):
    if df.empty:
        return df
    rename = {}
    lower_map = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for target, possible in aliases.items():
        if target in df.columns:
            continue
        for name in possible:
            if name in lower_map:
                rename[lower_map[name]] = target
                break
    return df.rename(columns=rename)


def heading(title, subtitle=""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


predictions = read_csv_safe(PREDICTIONS_FILE)
history = read_csv_safe(HISTORY_FILE)
hotspots = read_csv_safe(HOTSPOTS_FILE)
iot_data = read_csv_safe(DATA_DIR / "iot_noise_readings.csv")

predictions = normalize_columns(predictions, {
    "Audio_File": ["audio_file", "recording", "filename", "file"],
    "Predicted_Sound": ["predicted_sound", "predicted_class", "predicted_label", "prediction", "label", "class"],
    "Model_Probability_Percent": ["model_probability_percent", "probability_percent", "confidence", "confidence_percent"],
})
hotspots = normalize_columns(hotspots, {
    "Location": ["location", "area", "zone", "place"],
    "Latitude": ["latitude", "lat"],
    "Longitude": ["longitude", "lon", "lng"],
    "Total_Recordings": ["total_recordings", "recordings", "count"],
    "Priority_Events": ["priority_events", "event_count"],
    "Different_Sound_Classes": ["different_sound_classes", "sound_classes"],
    "Priority_Score_Percent": ["priority_score_percent", "priority_score", "score_percent"],
    "Rank": ["rank", "ranking"],
})

SCENARIOS = {
    "Road traffic": (58, 82, "Traffic activity"),
    "Traffic horn burst": (72, 94, "Car horn"),
    "Construction activity": (68, 96, "Construction noise"),
    "Public gathering": (54, 79, "Crowd activity"),
    "Quiet garden": (36, 54, "Ambient quiet"),
    "Emergency siren": (78, 99, "Siren"),
}
if "auralis_samples" not in st.session_state:
    random.seed(8)
    seed = []
    for scenario in ["Road traffic", "Public gathering", "Quiet garden", "Traffic horn burst", "Road traffic", "Construction activity"]:
        lo, hi, event = SCENARIOS[scenario]
        seed.append({"Time": datetime.now().strftime("%H:%M:%S"), "Scenario": scenario,
                     "Illustrative_dB": round(random.uniform(lo, hi), 1), "Event_Label": event,
                     "Source": "Simulated demo sample"})
    st.session_state.auralis_samples = seed
if "auralis_threshold" not in st.session_state:
    st.session_state.auralis_threshold = 70

st.sidebar.markdown("<div style='padding:8px 0 16px'><div style='color:#62ead8;letter-spacing:.18em;font-weight:800;font-size:.7rem'>URBAN SENSING LAB</div><div style='font-size:2rem;font-weight:900;color:#f5fbff'>AURALIS</div><div style='font-size:.8rem;color:#a7c0d0'>Acoustic intelligence control room</div></div>", unsafe_allow_html=True)
page = st.sidebar.radio("NAVIGATION", ["Mission Control", "Simulation Lab", "Audio Predictions", "Hotspot Atlas", "Project & Files"])
st.session_state.auralis_threshold = st.sidebar.slider("Demo alert threshold (dB)", 50, 90, int(st.session_state.auralis_threshold), help="Applies to simulated readings only.")
st.sidebar.caption("Recorded audio and simulated sensor data are kept separate.")

st.markdown("""
<div class="hero">
 <div class="kicker">SMART URBAN NOISE MONITORING · SOFTWARE PROTOTYPE</div>
 <div class="hero-title">Hear the city.<br><span>Understand the signal.</span></div>
 <div class="hero-copy">Explore urban sound classifications, test illustrative sensor scenarios, and inspect location-based event priorities in one acoustic-intelligence workspace.</div>
 <div><span class="pill">ML AUDIO CLASSIFICATION</span><span class="pill">SIMULATION SANDBOX</span><span class="pill">GIS HOTSPOT EXPLORER</span><span class="pill">LOCAL DEMO</span></div>
</div>
<div class="note"><b>Data transparency:</b> simulated dB values are for demonstration only, not physical sensor measurements. Map coordinates are manually assigned, not live GPS. Event priority is not an official noise-pollution index.</div>
""", unsafe_allow_html=True)

if page == "Mission Control":
    heading("Mission Control", "A live view of saved outputs and the interactive demonstration stream.")
    n_audio = len(predictions)
    n_classes = predictions["Predicted_Sound"].nunique() if n_audio and "Predicted_Sound" in predictions else 0
    n_zones = hotspots["Location"].nunique() if not hotspots.empty and "Location" in hotspots else 0
    n_events = int(pd.to_numeric(hotspots.get("Priority_Events", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not hotspots.empty else 0
    confidence = pd.to_numeric(predictions.get("Model_Probability_Percent", pd.Series(dtype=float)), errors="coerce").mean() if n_audio else float("nan")
    cols = st.columns(5)
    cols[0].metric("Audio files", n_audio)
    cols[1].metric("Sound classes", n_classes)
    cols[2].metric("Mapped zones", n_zones)
    cols[3].metric("Priority events", n_events)
    cols[4].metric("Mean confidence", f"{confidence:.1f}%" if pd.notna(confidence) else "—")
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        heading("Acoustic activity signature", "Count of predictions per detected sound class.")
        if n_audio and "Predicted_Sound" in predictions:
            st.bar_chart(predictions["Predicted_Sound"].fillna("unknown").astype(str).value_counts())
        else:
            st.info("Prediction data not found yet. Run your batch-prediction notebook cell to populate this panel.")
    with right:
        last = st.session_state.auralis_samples[-1] if st.session_state.auralis_samples else None
        if last:
            value = float(last["Illustrative_dB"])
            state = "CRITICAL" if value >= st.session_state.auralis_threshold + 15 else "ELEVATED" if value >= st.session_state.auralis_threshold else "WATCH" if value >= st.session_state.auralis_threshold - 10 else "CALM"
            st.metric("Latest simulated value", f"{value:.1f} dB*")
            st.markdown(f"**DEMO STATUS: {state}** · {last['Scenario']}")
            st.caption("*Illustrative simulation value, not a microphone reading.")
        if st.button("＋ Generate demo sample", use_container_width=True):
            scenario = random.choice(list(SCENARIOS))
            lo, hi, event = SCENARIOS[scenario]
            st.session_state.auralis_samples.append({"Time": datetime.now().strftime("%H:%M:%S"), "Scenario": scenario,
                "Illustrative_dB": round(random.uniform(lo, hi), 1), "Event_Label": event, "Source": "Generated demo sample"})
            st.rerun()
    heading("Latest audio predictions")
    if not predictions.empty:
        st.dataframe(predictions.head(10), use_container_width=True, hide_index=True)
        st.download_button("Download prediction CSV", predictions.to_csv(index=False).encode("utf-8"), "noise_predictions.csv", "text/csv")
    else:
        st.info(f"Expected prediction file: {PREDICTIONS_FILE}")

elif page == "Simulation Lab":
    heading("Simulation Lab", "Experiment with synthetic urban-noise scenarios and alert thresholds.")
    st.warning("SIMULATION ONLY: generated values are not readings from physical sensors.")
    c1, c2 = st.columns(2)
    scenario = c1.selectbox("Urban scenario", list(SCENARIOS))
    zone = c2.selectbox("Demo location label", ["Akurdi demo zone", "Dharmaraj Chowk", "DY Patil Road", "Garden", "Gurudwara", "Pradhikaran"])
    lo, hi, event = SCENARIOS[scenario]
    st.caption(f"Illustrative profile for {scenario}: {lo}–{hi} dB · event label: {event}")
    if st.button("⚡ Inject simulated reading", use_container_width=True):
        st.session_state.auralis_samples.append({"Time": datetime.now().strftime("%H:%M:%S"), "Scenario": scenario,
            "Illustrative_dB": round(random.uniform(lo, hi), 1), "Event_Label": event, "Source": f"Simulated at {zone}"})
        st.rerun()
    samples = pd.DataFrame(st.session_state.auralis_samples)
    if not samples.empty:
        latest = float(samples.iloc[-1]["Illustrative_dB"])
        state = "CRITICAL" if latest >= st.session_state.auralis_threshold + 15 else "ELEVATED" if latest >= st.session_state.auralis_threshold else "WATCH" if latest >= st.session_state.auralis_threshold - 10 else "CALM"
        a, b, c = st.columns(3)
        a.metric("Latest illustrative value", f"{latest:.1f} dB*")
        b.metric("Demo alert state", state)
        c.metric("Samples in session", len(samples))
        st.line_chart(samples["Illustrative_dB"].reset_index(drop=True))
        st.dataframe(samples.iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button("Download simulation log", samples.to_csv(index=False).encode("utf-8"), "simulated_noise_log.csv", "text/csv")
        if st.button("Reset simulation log"):
            st.session_state.auralis_samples = []
            st.rerun()

elif page == "Audio Predictions":
    heading("Audio Intelligence", "Inspect recorded-audio predictions generated by your trained classifier.")
    if not predictions.empty:
        if "Predicted_Sound" in predictions:
            choices = ["All classes"] + sorted(predictions["Predicted_Sound"].dropna().astype(str).unique())
            selected = st.selectbox("Filter by sound class", choices)
            shown = predictions if selected == "All classes" else predictions[predictions["Predicted_Sound"].astype(str) == selected]
        else:
            shown = predictions
        st.dataframe(shown, use_container_width=True, hide_index=True)
        st.download_button("Download current results", shown.to_csv(index=False).encode("utf-8"), "audio_predictions.csv", "text/csv")
    else:
        st.info(f"Prediction file not found: {PREDICTIONS_FILE}")
    st.markdown("#### Classify a new audio clip")
    uploaded = st.file_uploader("Upload WAV, MP3, FLAC, OGG, M4A or AAC", type=["wav", "mp3", "flac", "ogg", "m4a", "aac"])
    if uploaded and st.button("Analyse uploaded clip"):
        model_path = ROOT / "models" / "final_noise_classifier.pkl"
        encoder_path = ROOT / "models" / "label_encoder.pkl"
        if not model_path.exists() or not encoder_path.exists():
            st.error(f"Model files missing. Expected {model_path} and {encoder_path}")
        else:
            temp_path = None
            try:
                import tempfile, joblib, librosa, numpy as np
                suffix = Path(uploaded.name).suffix or ".wav"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded.getvalue())
                    temp_path = Path(tmp.name)
                audio, sr = librosa.load(str(temp_path), sr=22050, mono=True)
                if not len(audio):
                    raise ValueError("Empty or unreadable audio file.")
                mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40), axis=1)
                extra = [
                    np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
                    np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)),
                    np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)),
                    np.mean(librosa.feature.zero_crossing_rate(y=audio)),
                ]
                features = np.concatenate([mfcc, extra]).reshape(1, -1)
                model = joblib.load(model_path)
                encoder = joblib.load(encoder_path)
                raw = model.predict(features)[0]
                try:
                    predicted = encoder.inverse_transform([int(raw)])[0]
                except (ValueError, TypeError):
                    predicted = str(raw)
                st.success(f"Predicted sound class: {predicted}")
                if hasattr(model, "predict_proba"):
                    score = float(model.predict_proba(features).max()) * 100
                    st.metric("Model confidence estimate", f"{score:.2f}%")
                    st.caption("This is the model's confidence estimate for this clip, not overall accuracy.")
            except Exception as exc:
                st.error(f"Audio analysis failed: {exc}")
            finally:
                if temp_path:
                    try: temp_path.unlink(missing_ok=True)
                    except Exception: pass

elif page == "Hotspot Atlas":
    heading("Hotspot Atlas", "Location-level event priorities and the saved interactive map.")
    st.warning("Priority is based on selected predicted sound classes. It is not a calibrated decibel value or an official noise index.")
    if not hotspots.empty:
        cols = st.columns(3)
        cols[0].metric("Mapped locations", hotspots["Location"].nunique() if "Location" in hotspots else 0)
        cols[1].metric("Recordings represented", int(pd.to_numeric(hotspots.get("Total_Recordings", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()))
        best = pd.to_numeric(hotspots.get("Priority_Score_Percent", pd.Series(dtype=float)), errors="coerce").max()
        cols[2].metric("Highest event score", f"{best:.1f}%" if pd.notna(best) else "—")
        st.dataframe(hotspots, use_container_width=True, hide_index=True)
        if "Location" in hotspots and "Priority_Score_Percent" in hotspots:
            st.bar_chart(hotspots.set_index("Location")[["Priority_Score_Percent"]])
        st.download_button("Download hotspot CSV", hotspots.to_csv(index=False).encode("utf-8"), "hotspot_priority.csv", "text/csv")
    else:
        st.info(f"Hotspot CSV not found: {HOTSPOTS_FILE}")
    st.markdown("#### Interactive GIS map")
    if MAP_FILE.exists():
        try:
            components.html(MAP_FILE.read_text(encoding="utf-8"), height=600, scrolling=True)
            st.download_button("Download map HTML", MAP_FILE.read_bytes(), "noise_hotspot_map.html", "text/html")
        except Exception as exc:
            st.error(f"Could not display map: {exc}")
            st.write(str(MAP_FILE))
    else:
        st.info(f"Map not found: {MAP_FILE}")

elif page == "Project & Files":
    heading("Project & Files", "Pipeline overview and local output readiness.")
    left, right = st.columns(2)
    with left:
        st.markdown("### Audio pipeline")
        st.write("Recorded clip → MFCC and spectral features → trained classifier → predicted sound label.")
        st.markdown("### Event priority")
        st.write("Counts selected predicted sound classes by mapped location to calculate a preliminary priority score.")
    with right:
        st.markdown("### GIS layer")
        st.write("The Folium map uses coordinates stored in the project; it does not access live GPS.")
        st.markdown("### Simulation layer")
        st.write("Generated scenario values let you demonstrate thresholds, alerts, and trends without physical sensors.")
    file_checks = [
        ("Batch audio predictions", PREDICTIONS_FILE),
        ("Event history", HISTORY_FILE),
        ("Hotspot priorities", HOTSPOTS_FILE),
        ("Interactive map", MAP_FILE),
        ("Trained classifier", ROOT / "models" / "final_noise_classifier.pkl"),
        ("Label encoder", ROOT / "models" / "label_encoder.pkl"),
        ("Simulated IoT dataset", DATA_DIR / "iot_noise_readings.csv"),
    ]
    status = pd.DataFrame([{"File": label, "Status": "READY" if path.exists() else "NOT FOUND", "Path": str(path)} for label, path in file_checks])
    st.dataframe(status, use_container_width=True, hide_index=True)

st.divider()
st.caption("AURALIS · Academic software prototype · Recorded audio + explicitly labelled demo simulation · No live GPS or physical sensor claim")
