# ================================================================
# AURALIS | ACOUSTIC DIGITAL TWIN
# Smart Urban Noise Monitoring System
# Interactive software-only project
# ================================================================

from pathlib import Path
from datetime import datetime, timedelta
import html
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# ================================================================
# 1. APPLICATION CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="AURALIS | Acoustic Digital Twin",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

PREDICTION_FILE = DATA_DIR / "akurdi_batch_predictions.csv"
HISTORY_FILE = DATA_DIR / "akurdi_real_event_history.csv"
HOTSPOT_FILE = DATA_DIR / "akurdi_real_hotspot_priority.csv"
MAP_FILE = DATA_DIR / "akurdi_real_audio_hotspots_map.html"
MODEL_FILE = MODELS_DIR / "final_noise_classifier.pkl"
ENCODER_FILE = MODELS_DIR / "label_encoder.pkl"


# ================================================================
# 2. FUTURISTIC INTERFACE
# ================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg: #060e19;
    --panel: #0d1b2b;
    --line: #203c50;
    --cyan: #4ae5dc;
    --blue: #69aaff;
    --purple: #aa91ff;
    --muted: #8da8bd;
}

.stApp {
    background:
        radial-gradient(ellipse at 80% 0%, #12334a 0%, transparent 34%),
        radial-gradient(ellipse at 0% 40%, #102638 0%, transparent 32%),
        #060e19;
    color: #ecf7ff;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1550px;
    padding-top: 1.35rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #081522, #0a1828);
    border-right: 1px solid #20394d;
}

[data-testid="stSidebar"] * {
    color: #dceaf7;
}

h1, h2, h3, h4 {
    color: #f2f8ff !important;
    letter-spacing: -.4px;
}

p, label, [data-testid="stCaptionContainer"] {
    color: #b2c5d6;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 30px 32px;
    border-radius: 21px;
    border: 1px solid #285469;
    background: linear-gradient(115deg, #103248, #0c1d2c 58%, #113039);
    margin-bottom: 23px;
    box-shadow: 0 12px 48px rgba(0,0,0,.24);
}

.hero:after {
    content: "";
    position: absolute;
    width: 210px;
    height: 210px;
    right: -45px;
    top: -125px;
    border: 1px solid rgba(74,229,220,.35);
    border-radius: 50%;
    box-shadow: 0 0 0 25px rgba(74,229,220,.035),
                0 0 0 50px rgba(74,229,220,.025);
}

.eyebrow {
    font-family: 'Space Mono', monospace;
    color: var(--cyan);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 3px;
}

.hero-title {
    color: #fff;
    font-size: clamp(32px, 4.2vw, 48px);
    line-height: 1.15;
    font-weight: 800;
    letter-spacing: 5px;
    margin-top: 10px;
}

.hero-desc {
    color: #bed2e0;
    font-size: 14px;
    line-height: 1.8;
    margin-top: 9px;
    max-width: 800px;
}

.section-tag {
    font-family: 'Space Mono', monospace;
    color: var(--cyan);
    font-size: 10px;
    letter-spacing: 2.2px;
    margin-top: 7px;
    margin-bottom: 6px;
}

.panel {
    background: linear-gradient(145deg, #102337, #0b1928);
    border: 1px solid #254357;
    border-radius: 15px;
    padding: 18px;
}

.panel-label {
    color: #93adbf;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.8px;
}

.panel-value {
    color: var(--cyan);
    font-size: 26px;
    font-weight: 800;
    margin-top: 8px;
}

.panel-sub {
    color: #b5c9d9;
    font-size: 12px;
    line-height: 1.7;
    margin-top: 6px;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #10273a, #0b1928);
    border: 1px solid #254357;
    border-radius: 14px;
    padding: 15px;
}

div[data-testid="stMetricLabel"] {
    color: #9db9cb;
}

div[data-testid="stMetricValue"] {
    color: var(--cyan);
}

.stButton > button {
    background: linear-gradient(115deg, #123e4d, #17414a);
    border: 1px solid #347b83;
    color: #f4ffff;
    border-radius: 10px;
    font-weight: 700;
    min-height: 42px;
    transition: .2s ease;
}

.stButton > button:hover {
    background: #1b5360;
    border-color: var(--cyan);
    color: #fff;
    box-shadow: 0 0 16px rgba(74,229,220,.12);
}

.stDownloadButton > button {
    border-radius: 10px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #213c51;
    border-radius: 11px;
}

hr {
    border-color: #203b4f;
}

.stTabs [aria-selected="true"] {
    color: var(--cyan) !important;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ================================================================
# 3. PROJECT DATA
# ================================================================

ZONES = {
    "Dharmaraj Chowk": {
        "lat": 18.6497,
        "lon": 73.7617,
        "type": "Road junction",
        "base": 76
    },
    "DY Patil Road": {
        "lat": 18.650890,
        "lon": 73.778496,
        "type": "Road corridor",
        "base": 79
    },
    "Garden": {
        "lat": 18.6548,
        "lon": 73.7663,
        "type": "Green space",
        "base": 49
    },
    "Gurudwara": {
        "lat": 18.6452,
        "lon": 73.7644,
        "type": "Community area",
        "base": 61
    },
    "Pradhikaran": {
        "lat": 18.6483,
        "lon": 73.7647,
        "type": "Residential area",
        "base": 57
    }
}

SCENARIOS = {
    "Peak-hour traffic": {
        "base": 82,
        "variation": 9,
        "event": "Traffic surge"
    },
    "Construction activity": {
        "base": 88,
        "variation": 7,
        "event": "Construction activity"
    },
    "School dispersal": {
        "base": 75,
        "variation": 10,
        "event": "High public activity"
    },
    "Residential quiet period": {
        "base": 48,
        "variation": 6,
        "event": "Quiet-zone observation"
    },
    "Emergency vehicle": {
        "base": 91,
        "variation": 5,
        "event": "Emergency-siren scenario"
    },
    "Night-time road": {
        "base": 56,
        "variation": 7,
        "event": "Reduced road activity"
    }
}

INTERVENTIONS = {
    "Traffic-flow adjustment": {
        "reduction": 5,
        "note": "Illustrative traffic-management scenario."
    },
    "Construction scheduling": {
        "reduction": 4,
        "note": "Illustrative change in construction activity."
    },
    "Noise barrier": {
        "reduction": 7,
        "note": "Illustrative barrier-effect scenario."
    },
    "Quiet-zone policy": {
        "reduction": 3,
        "note": "Illustrative reduction in selected activity."
    },
    "Route diversion": {
        "reduction": 6,
        "note": "Illustrative traffic-diversion scenario."
    }
}


def read_csv_safely(path):
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def find_column(df, candidates):
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]

    return None


predictions = read_csv_safely(PREDICTION_FILE)
history = read_csv_safely(HISTORY_FILE)
hotspots = read_csv_safely(HOTSPOT_FILE)

sound_col = find_column(
    predictions,
    [
        "Predicted_Sound", "predicted_class",
        "predicted_label", "prediction", "label", "class"
    ]
)

audio_col = find_column(
    predictions,
    ["Audio_File", "Recording", "filename", "file_name"]
)


# ================================================================
# 4. SIMULATED DIGITAL TWIN ENGINE
# ================================================================

def get_status(level, threshold):
    if level >= threshold + 12:
        return "CRITICAL"
    if level >= threshold:
        return "ELEVATED"
    return "NORMAL"


def generate_initial_log():
    rng = np.random.default_rng(31)
    rows = []
    now = datetime.now()

    zone_names = list(ZONES)
    scenario_names = list(SCENARIOS)

    for cycle in range(8):
        moment = now - timedelta(minutes=(7-cycle)*5)

        for index, zone in enumerate(zone_names):
            scenario = scenario_names[(cycle + index) % len(scenario_names)]
            config = SCENARIOS[scenario]

            value = int(np.clip(
                rng.normal(
                    (config["base"] * 0.65 + ZONES[zone]["base"] * 0.35),
                    config["variation"] / 2
                ),
                35, 110
            ))

            rows.append({
                "Cycle": cycle + 1,
                "Time": moment.strftime("%H:%M:%S"),
                "Zone": zone,
                "Scenario": scenario,
                "Event": config["event"],
                "Simulated_Level": value,
                "Source": "SIMULATION"
            })

    result = pd.DataFrame(rows)
    result["Status"] = result["Simulated_Level"].map(
        lambda x: get_status(x, 75)
    )
    return result


if "twin_log" not in st.session_state:
    st.session_state.twin_log = generate_initial_log()

if "cycle_number" not in st.session_state:
    st.session_state.cycle_number = 8


# ================================================================
# 5. VISUAL COMPONENTS
# ================================================================

def metric_card(label, value, note):
    st.markdown(f"""
    <div class="panel">
        <div class="panel-label">{html.escape(str(label))}</div>
        <div class="panel-value">{html.escape(str(value))}</div>
        <div class="panel-sub">{html.escape(str(note))}</div>
    </div>
    """, unsafe_allow_html=True)


def render_radar(level, label="ACOUSTIC FIELD"):
    intensity = min(max(int(level), 35), 110)

    markup = """
    <html><head><style>
    * { box-sizing:border-box; }
    body {margin:0;background:transparent;font-family:Arial;color:#dffaff;}
    .wrap {
        height:300px;position:relative;overflow:hidden;
        background:linear-gradient(145deg,#0d2031,#091624);
        border:1px solid #25475b;border-radius:17px;
        display:flex;justify-content:center;align-items:center;
    }
    .radar {
        width:220px;height:220px;border-radius:50%;position:relative;
        border:1px solid #28616b;
        background:radial-gradient(circle,
        transparent 0 20%,rgba(74,229,220,.07) 21% 22%,
        transparent 23% 42%,rgba(74,229,220,.07) 43% 44%,
        transparent 45% 64%,rgba(74,229,220,.07) 65% 66%,transparent 67%);
        box-shadow:0 0 28px rgba(74,229,220,.09);
    }
    .radar:before {
        content:"";position:absolute;left:50%;top:0;width:1px;height:100%;
        background:rgba(74,229,220,.27);
    }
    .radar:after {
        content:"";position:absolute;left:0;top:50%;height:1px;width:100%;
        background:rgba(74,229,220,.27);
    }
    .sweep {
        position:absolute;inset:0;border-radius:50%;
        background:conic-gradient(from 0deg,transparent 0deg,
        transparent 285deg,rgba(74,229,220,.02) 300deg,
        rgba(74,229,220,.48) 359deg,transparent 360deg);
        animation:spin 5s linear infinite;
    }
    .ring {
        position:absolute;inset:30px;border:1px solid rgba(74,229,220,.26);
        border-radius:50%;animation:pulse 2.8s ease-out infinite;
    }
    .ring2 {inset:56px;animation-delay:1s;}
    .center {
        position:absolute;top:calc(50% - 5px);left:calc(50% - 5px);
        width:10px;height:10px;border-radius:50%;
        background:#5ef8e5;box-shadow:0 0 18px #5ef8e5;
    }
    .dot {
        position:absolute;width:7px;height:7px;border-radius:50%;
        background:#a991ff;box-shadow:0 0 12px #a991ff;
    }
    .d1 {left:66%;top:26%;}
    .d2 {left:25%;top:64%;background:#57e9db;}
    .d3 {left:77%;top:66%;background:#ffbe72;}
    .meta {
        position:absolute;font-family:monospace;font-size:9px;
        color:#8bb8c6;letter-spacing:1px;line-height:1.7;
    }
    .left {top:16px;left:16px;}
    .right {right:16px;top:16px;text-align:right;}
    .value {
        position:absolute;bottom:17px;font-size:15px;
        text-align:center;color:#5ceadd;font-weight:bold;
    }
    .value small {display:block;font-size:8px;letter-spacing:2px;color:#95b2c2;margin-top:3px;}
    @keyframes spin {to{transform:rotate(360deg);}}
    @keyframes pulse {
        0%{transform:scale(.75);opacity:.8;}
        100%{transform:scale(1.2);opacity:0;}
    }
    </style></head><body>
    <div class="wrap">
      <div class="meta left">ACOUSTIC FIELD<br>DEMO SCAN</div>
      <div class="meta right">DIGITAL TWIN<br>SIMULATED SIGNAL</div>
      <div class="radar">
        <div class="sweep"></div><div class="ring"></div>
        <div class="ring ring2"></div><div class="center"></div>
        <div class="dot d1"></div><div class="dot d2"></div><div class="dot d3"></div>
      </div>
      <div class="value">LEVEL_VALUE
        <small>ILLUSTRATIVE LEVEL / NOT MEASURED</small>
      </div>
    </div></body></html>
    """

    markup = markup.replace("LEVEL_VALUE", str(intensity))
    components.html(markup, height=305)


def schematic_network():
    markup = """
    <html><head><style>
      body {margin:0;background:transparent;font-family:Arial;color:#e5f5ff;}
      .box {background:#0b1a29;border:1px solid #25475a;border-radius:15px;padding:18px;min-height:250px;}
      .tag {font-family:monospace;font-size:10px;color:#59e4d5;letter-spacing:2px;}
      .title {font-size:17px;font-weight:bold;margin:9px 0;color:#f3f9ff;}
      .city {position:relative;height:165px;margin-top:8px;
       background-image:linear-gradient(#1c364855 1px,transparent 1px),
       linear-gradient(90deg,#1c364855 1px,transparent 1px);
       background-size:25px 25px;border-radius:8px;}
      .road {position:absolute;background:#204053;}
      .r1 {left:0;right:0;height:5px;top:43%;}
      .r2 {top:0;bottom:0;width:5px;left:52%;}
      .node {position:absolute;width:12px;height:12px;border-radius:50%;
       background:#50ead8;border:2px solid #c3fff5;box-shadow:0 0 12px #50ead8;}
      .n1{left:20%;top:22%}.n2{left:70%;top:30%}
      .n3{left:40%;top:61%}.n4{left:80%;top:72%}.n5{left:13%;top:69%}
      .legend {display:flex;gap:14px;flex-wrap:wrap;margin-top:12px;font-size:10px;color:#a4bdce}
      .dot {display:inline-block;width:7px;height:7px;border-radius:50%;background:#50ead8;margin-right:5px}
      .dot2 {background:#a991ff}.dot3{background:#ffbd73}
      .note {font-size:10px;color:#7fa3b7;margin-top:12px;}
    </style></head><body>
    <div class="box">
      <div class="tag">NETWORK TOPOLOGY / DEMO</div>
      <div class="title">Urban Monitoring Model</div>
      <div class="city">
       <div class="road r1"></div><div class="road r2"></div>
       <div class="node n1"></div><div class="node n2"></div>
       <div class="node n3"></div><div class="node n4"></div>
       <div class="node n5"></div>
      </div>
      <div class="legend">
       <span><i class="dot"></i>Configured zone</span>
       <span><i class="dot dot2"></i>Analysis module</span>
       <span><i class="dot dot3"></i>Scenario event</span>
      </div>
      <div class="note">Schematic visualization only — not a live sensor network or a geographic map.</div>
    </div></body></html>
    """
    components.html(markup, height=315)


def next_simulation_cycle(zone, scenario, intensity, threshold):
    rng = np.random.default_rng()

    rows = []
    timestamp = datetime.now().strftime("%H:%M:%S")
    cycle_id = st.session_state.cycle_number + 1

    for place, info in ZONES.items():
        if place == zone:
            chosen_scenario = scenario
            config = SCENARIOS[scenario]
            base = config["base"]
            variation = config["variation"]
            base = base + (intensity - 5) * 3
        else:
            chosen_scenario = "Background urban activity"
            base = info["base"]
            variation = 6

        level = int(np.clip(
            rng.normal(base, variation / 2),
            35, 110
        ))

        if place == zone:
            event = SCENARIOS[scenario]["event"]
        else:
            event = "Background activity"

        rows.append({
            "Cycle": cycle_id,
            "Time": timestamp,
            "Zone": place,
            "Scenario": chosen_scenario,
            "Event": event,
            "Simulated_Level": level,
            "Status": get_status(level, threshold),
            "Source": "SIMULATION"
        })

    fresh = pd.DataFrame(rows)

    st.session_state.twin_log = pd.concat(
        [st.session_state.twin_log, fresh],
        ignore_index=True
    ).tail(250).reset_index(drop=True)

    st.session_state.cycle_number = cycle_id


# ================================================================
# 6. SIDEBAR NAVIGATION
# ================================================================

with st.sidebar:
    st.markdown(
        '<div class="eyebrow">ACOUSTIC INTELLIGENCE</div>',
        unsafe_allow_html=True
    )
    st.title("◉ AURALIS")
    st.caption("Urban Acoustic Digital Twin")

    st.divider()

    page = st.radio(
        "COMMAND NAVIGATION",
        [
            "Mission Control",
            "Live Simulation",
            "Scenario Laboratory",
            "Audio Intelligence",
            "Hotspot Atlas",
            "Intervention Studio",
            "System Architecture"
        ]
    )

    st.divider()

    st.markdown("**PROJECT MODULES**")
    st.write("🟢 Interactive dashboard")
    st.write(
        "🟢 ML model found"
        if MODEL_FILE.exists()
        else "🟡 ML model not found"
    )
    st.write(
        "🟢 Prediction data found"
        if not predictions.empty
        else "🟡 Prediction CSV missing"
    )
    st.write(
        "🟢 GIS file found"
        if MAP_FILE.exists()
        else "🟡 GIS map missing"
    )

    st.divider()
    st.caption("SOFTWARE-ONLY PROTOTYPE")
    st.caption("Generated readings are labelled as simulation.")


# ================================================================
# 7. HERO HEADER
# ================================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">SMART CITY TECHNOLOGY / ACOUSTIC COMPUTING</div>
    <div class="hero-title">A U R A L I S</div>
    <div class="hero-desc">
        Interactive acoustic intelligence for recorded-sound classification,
        urban-event simulation, spatial analysis and what-if planning.
        <br><br>
        <span style="color:#4ae5dc">◉ DIGITAL TWIN READY</span>
        &nbsp;&nbsp; / &nbsp;&nbsp; SOFTWARE DEMONSTRATION MODE
    </div>
</div>
""", unsafe_allow_html=True)


# ================================================================
# PAGE 1 — MISSION CONTROL
# ================================================================

if page == "Mission Control":

    st.markdown(
        '<div class="section-tag">01 / COMMAND CENTRE</div>',
        unsafe_allow_html=True
    )
    st.header("Acoustic Mission Control")

    log = st.session_state.twin_log
    latest = log.iloc[-1]

    total_audio = len(predictions)
    total_classes = (
        predictions[sound_col].nunique()
        if sound_col and not predictions.empty else 0
    )
    total_zones = (
        hotspots["Location"].nunique()
        if not hotspots.empty and "Location" in hotspots.columns
        else len(ZONES)
    )
    high_demo = int((log["Status"] != "NORMAL").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RECORDED AUDIO FILES", total_audio)
    c2.metric("SOUND CLASSES", total_classes)
    c3.metric("MAPPED DATA LOCATIONS", total_zones)
    c4.metric("SIMULATED ALERT RECORDS", high_demo)

    left, right = st.columns([1, 1])

    with left:
        render_radar(latest["Simulated_Level"])
    with right:
        schematic_network()

    st.divider()

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Acoustic Activity Timeline")
        st.line_chart(
            log.set_index("Time")["Simulated_Level"],
            height=280
        )
        st.caption(
            "Synthetic readings are displayed to demonstrate the analytics workflow."
        )

    with right:
        st.subheader("Activity by Zone")
        mean_levels = log.groupby("Zone")["Simulated_Level"].mean()
        st.bar_chart(mean_levels, height=280)

    st.divider()
    st.subheader("Recorded Model Predictions")

    if not predictions.empty:
        if sound_col:
            distribution = (
                predictions[sound_col]
                .fillna("Unknown")
                .astype(str)
                .value_counts()
            )

            left, right = st.columns([1, 1])

            with left:
                st.markdown("**Sound-class distribution**")
                st.bar_chart(distribution)

            with right:
                st.markdown("**Latest prediction records**")
                st.dataframe(
                    predictions.tail(8),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.dataframe(
                predictions,
                use_container_width=True,
                hide_index=True
            )

        st.download_button(
            "Download model prediction records",
            data=predictions.to_csv(index=False).encode("utf-8"),
            file_name="auralis_predictions.csv",
            mime="text/csv"
        )
    else:
        st.info(
            "Actual audio prediction records will appear here after "
            "running the batch-prediction cell in your notebook."
        )

    st.warning(
        "Simulated readings are not physical sensor measurements. "
        "Recorded-audio predictions are separate from the simulation."
    )


# ================================================================
# PAGE 2 — LIVE SIMULATION
# ================================================================

elif page == "Live Simulation":

    st.markdown(
        '<div class="section-tag">02 / ACOUSTIC DIGITAL TWIN</div>',
        unsafe_allow_html=True
    )
    st.header("Live Simulation Chamber")

    st.write(
        "Select an urban zone, create a hypothetical event, and advance "
        "the monitoring model by one cycle."
    )

    st.info(
        "DEMO MODE: the simulation does not receive real-time input from "
        "a sound-level meter or a physical sensor."
    )

    left, right = st.columns([1, 1])

    with left:
        selected_zone = st.selectbox(
            "Target zone",
            list(ZONES.keys())
        )

        selected_scenario = st.selectbox(
            "Urban event to inject",
            list(SCENARIOS.keys())
        )

        intensity = st.slider(
            "Event intensity index",
            min_value=1,
            max_value=10,
            value=6
        )

    with right:
        threshold = st.slider(
            "Illustrative alert threshold",
            min_value=45,
            max_value=105,
            value=75
        )

        st.markdown("#### Selected zone profile")

        st.write(f"**Zone:** {selected_zone}")
        st.write(f"**Environment:** {ZONES[selected_zone]['type']}")
        st.write(f"**Scenario:** {selected_scenario}")

        st.caption(
            "Intensity index is a simulation control, not a standardized "
            "environmental measurement."
        )

    if st.button("▶ ADVANCE SIMULATION CYCLE", use_container_width=True):
        next_simulation_cycle(
            selected_zone,
            selected_scenario,
            intensity,
            threshold
        )
        st.success(
            f"Simulation cycle {st.session_state.cycle_number} completed."
        )

    log = st.session_state.twin_log.copy()

    log["Status"] = log["Simulated_Level"].map(
        lambda value: get_status(value, threshold)
    )

    latest = log.iloc[-1]
    latest_level = int(latest["Simulated_Level"])

    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("LATEST SIMULATED LEVEL", f"{latest_level} dB*")
    c2.metric("ALERT THRESHOLD", f"{threshold} dB*")
    c3.metric(
        "LATEST DEMO STATE",
        get_status(latest_level, threshold)
    )

    if latest_level >= threshold:
        st.error(
            "SIMULATION ALERT: the displayed value exceeds the "
            "illustrative threshold."
        )
    else:
        st.success(
            "SIMULATION NORMAL: the displayed value is below the "
            "illustrative threshold."
        )

    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("Monitoring Timeline")
        st.line_chart(log.set_index("Time")["Simulated_Level"])

    with right:
        st.subheader("Latest Activity by Zone")
        latest_cycle = log["Cycle"].max()
        last_readings = log[log["Cycle"] == latest_cycle]

        st.bar_chart(
            last_readings.set_index("Zone")["Simulated_Level"]
        )

    st.subheader("Simulation Event Log")
    st.dataframe(
        log.sort_index(ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download simulation log",
        data=log.to_csv(index=False).encode("utf-8"),
        file_name="auralis_simulation_log.csv",
        mime="text/csv"
    )

    st.caption(
        "*Illustrative levels only. Values are not calibrated decibel readings."
    )


# ================================================================
# PAGE 3 — SCENARIO LABORATORY
# ================================================================

elif page == "Scenario Laboratory":

    st.markdown(
        '<div class="section-tag">03 / SCENARIO DESIGN</div>',
        unsafe_allow_html=True
    )
    st.header("Urban Scenario Laboratory")

    st.write(
        "Explore how different hypothetical urban scenarios change "
        "the simulated profile."
    )

    chosen_zones = st.multiselect(
        "Select zones to include",
        list(ZONES.keys()),
        default=list(ZONES.keys())
    )

    chosen_scenarios = st.multiselect(
        "Select scenarios to compare",
        list(SCENARIOS.keys()),
        default=[
            "Peak-hour traffic",
            "Construction activity",
            "Residential quiet period"
        ]
    )

    seed = st.number_input(
        "Scenario seed for repeatable results",
        min_value=1,
        max_value=999999,
        value=42,
        step=1
    )

    threshold = st.slider(
        "Illustrative scenario alert threshold",
        min_value=45,
        max_value=105,
        value=75,
        key="scenario_threshold"
    )

    if st.button("GENERATE SCENARIO COMPARISON"):
        if not chosen_zones or not chosen_scenarios:
            st.warning("Select at least one zone and one scenario.")
        else:
            rng = np.random.default_rng(int(seed))
            rows = []

            for scenario in chosen_scenarios:
                config = SCENARIOS[scenario]

                for zone in chosen_zones:
                    value = int(np.clip(
                        rng.normal(
                            (config["base"] * 0.7
                             + ZONES[zone]["base"] * 0.3),
                            config["variation"] / 2
                        ),
                        35,
                        110
                    ))

                    rows.append({
                        "Zone": zone,
                        "Scenario": scenario,
                        "Simulated_Level": value,
                        "Status": get_status(value, threshold),
                        "Source": "SIMULATION"
                    })

            st.session_state.scenario_results = pd.DataFrame(rows)

    if "scenario_results" in st.session_state:
        results = st.session_state.scenario_results

        st.divider()
        st.subheader("Scenario Comparison")

        chart_data = results.pivot_table(
            index="Zone",
            columns="Scenario",
            values="Simulated_Level",
            aggfunc="mean"
        )

        st.bar_chart(chart_data)

        st.subheader("Generated Scenario Records")
        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download scenario comparison",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="auralis_scenario_comparison.csv",
            mime="text/csv"
        )

    else:
        st.info(
            "Choose the scenarios and zones, then generate a comparison."
        )

    st.warning(
        "Scenario outputs are synthetic and demonstrate how comparative "
        "analytics could operate. They are not observations or forecasts "
        "of actual future sound levels."
    )


# ================================================================
# PAGE 4 — AUDIO INTELLIGENCE
# ================================================================

elif page == "Audio Intelligence":

    st.markdown(
        '<div class="section-tag">04 / MACHINE LEARNING</div>',
        unsafe_allow_html=True
    )
    st.header("Audio Intelligence Lab")

    st.write(
        "Inspect predictions from your saved model or test a compatible "
        "audio file using the trained classifier."
    )

    tab1, tab2 = st.tabs([
        "Prediction Explorer",
        "Classify New Audio"
    ])

    with tab1:
        if not predictions.empty:

            filtered = predictions.copy()

            if sound_col:
                classes = sorted(
                    filtered[sound_col].dropna().astype(str).unique()
                )

                selected_classes = st.multiselect(
                    "Filter predicted classes",
                    classes,
                    default=classes
                )

                filtered = filtered[
                    filtered[sound_col].astype(str).isin(selected_classes)
                ]

            st.dataframe(
                filtered,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Export prediction records",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="auralis_filtered_predictions.csv",
                mime="text/csv"
            )

        else:
            st.info(
                "The prediction CSV was not found. Generate it from "
                "your notebook first."
            )

    with tab2:
        uploaded = st.file_uploader(
            "Upload a sound recording",
            type=["wav", "mp3", "flac", "ogg", "m4a"]
        )

        if uploaded:
            st.audio(uploaded)

        if uploaded and st.button("ANALYZE UPLOADED AUDIO"):
            temporary_path = None

            try:
                import joblib
                import librosa

                if not MODEL_FILE.exists() or not ENCODER_FILE.exists():
                    raise FileNotFoundError(
                        "The saved model or label encoder was not found."
                    )

                model = joblib.load(MODEL_FILE)
                encoder = joblib.load(ENCODER_FILE)

                with tempfile.NamedTemporaryFile(
                    suffix=Path(uploaded.name).suffix,
                    delete=False
                ) as temp:
                    temp.write(uploaded.getvalue())
                    temporary_path = Path(temp.name)

                audio, sr = librosa.load(
                    str(temporary_path),
                    sr=22050,
                    mono=True
                )

                if len(audio) == 0:
                    raise ValueError("Audio file contains no samples.")

                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=sr,
                    n_mfcc=40
                )

                features = np.concatenate([
                    np.mean(mfcc, axis=1),
                    [
                        np.mean(
                            librosa.feature.spectral_centroid(y=audio, sr=sr)
                        ),
                        np.mean(
                            librosa.feature.spectral_bandwidth(y=audio, sr=sr)
                        ),
                        np.mean(
                            librosa.feature.spectral_rolloff(y=audio, sr=sr)
                        ),
                        np.mean(
                            librosa.feature.zero_crossing_rate(y=audio)
                        )
                    ]
                ])

                if len(features) != 44:
                    raise ValueError(
                        f"Expected 44 features, extracted {len(features)}."
                    )

                X = features.reshape(1, -1)
                raw_prediction = model.predict(X)[0]

                try:
                    predicted_label = encoder.inverse_transform(
                        [int(raw_prediction)]
                    )[0]
                except (ValueError, TypeError):
                    predicted_label = str(raw_prediction)

                st.success("Audio prediction completed.")

                left, right = st.columns(2)

                left.metric(
                    "PREDICTED SOUND",
                    str(predicted_label)
                )

                if hasattr(model, "predict_proba"):
                    probability = model.predict_proba(X)[0]
                    right.metric(
                        "MODEL CONFIDENCE ESTIMATE",
                        f"{float(np.max(probability)) * 100:.2f}%"
                    )

                st.caption(
                    "Correctness depends on consistent feature extraction "
                    "and compatibility with the model's training pipeline."
                )

            except Exception as exc:
                st.error(f"Audio analysis failed: {exc}")

            finally:
                if temporary_path:
                    try:
                        temporary_path.unlink(missing_ok=True)
                    except Exception:
                        pass


# ================================================================
# PAGE 5 — HOTSPOT ATLAS
# ================================================================

elif page == "Hotspot Atlas":

    st.markdown(
        '<div class="section-tag">05 / GEOSPATIAL ANALYTICS</div>',
        unsafe_allow_html=True
    )
    st.header("Urban Hotspot Atlas")

    if MAP_FILE.exists():
        st.subheader("Recorded-Audio GIS Map")

        try:
            components.html(
                MAP_FILE.read_text(encoding="utf-8"),
                height=590,
                scrolling=True
            )
        except Exception as exc:
            st.warning(f"Unable to display the saved map: {exc}")

    else:
        st.info(
            "The recorded-audio GIS HTML is not available. "
            "Showing the configured project locations as a fallback."
        )

        points = pd.DataFrame([
            {
                "Location": name,
                "Latitude": info["lat"],
                "Longitude": info["lon"]
            }
            for name, info in ZONES.items()
        ])

        st.map(
            points,
            latitude="Latitude",
            longitude="Longitude"
        )

    st.divider()
    st.subheader("Hotspot Priority Analysis")

    if not hotspots.empty:
        st.dataframe(
            hotspots,
            use_container_width=True,
            hide_index=True
        )

        if (
            "Location" in hotspots.columns
            and "Priority_Score_Percent" in hotspots.columns
        ):
            chart = hotspots[["Location", "Priority_Score_Percent"]].copy()

            chart["Priority_Score_Percent"] = pd.to_numeric(
                chart["Priority_Score_Percent"],
                errors="coerce"
            )

            chart = chart.dropna()

            if not chart.empty:
                st.bar_chart(
                    chart.set_index("Location")["Priority_Score_Percent"]
                )

        st.download_button(
            "Download hotspot analysis",
            data=hotspots.to_csv(index=False).encode("utf-8"),
            file_name="auralis_hotspot_priority.csv",
            mime="text/csv"
        )

    else:
        st.warning(
            "No saved hotspot priority table found. Generate it using "
            "your notebook before exploring the recorded-data analysis."
        )

    st.warning(
        "Map coordinates are manually assigned and must be verified. "
        "The map does not show live GPS, and event-priority scores are "
        "not calibrated noise measurements or legal-risk assessments."
    )


# ================================================================
# PAGE 6 — INTERVENTION STUDIO
# ================================================================

elif page == "Intervention Studio":

    st.markdown(
        '<div class="section-tag">06 / WHAT-IF PLANNING</div>',
        unsafe_allow_html=True
    )
    st.header("Urban Intervention Studio")

    st.write(
        "Compare a baseline scenario with a hypothetical intervention "
        "to demonstrate decision-support analysis."
    )

    st.info(
        "The intervention effect is assumed by the user for simulation. "
        "It is not a validated prediction of real-world improvement."
    )

    zone = st.selectbox(
        "Choose an urban zone",
        list(ZONES.keys()),
        key="intervention_zone"
    )

    scenario = st.selectbox(
        "Choose an urban scenario",
        list(SCENARIOS.keys()),
        key="intervention_scenario"
    )

    strategy = st.selectbox(
        "Choose a hypothetical intervention",
        list(INTERVENTIONS.keys())
    )

    baseline = st.slider(
        "Baseline simulated level",
        min_value=35,
        max_value=110,
        value=SCENARIOS[scenario]["base"]
    )

    assumed_reduction = st.slider(
        "Assumed reduction for the what-if calculation",
        min_value=0,
        max_value=20,
        value=INTERVENTIONS[strategy]["reduction"]
    )

    after = max(30, baseline - assumed_reduction)

    st.divider()

    a, b, c = st.columns(3)
    a.metric("BASELINE SCENARIO", f"{baseline} dB*")
    b.metric("ASSUMED CHANGE", f"-{assumed_reduction} dB*")
    c.metric("HYPOTHETICAL RESULT", f"{after} dB*")

    before_after = pd.DataFrame(
        {"Illustrative level": [baseline, after]},
        index=["Baseline", "After intervention"]
    )

    st.subheader("Before and After Comparison")
    st.bar_chart(before_after)

    st.markdown(f"""
    <div class="panel">
        <div class="panel-label">SELECTED INTERVENTION</div>
        <div class="panel-value" style="font-size:23px">
            {html.escape(strategy)}
        </div>
        <div class="panel-sub">
            Zone: {html.escape(zone)}<br>
            Scenario: {html.escape(scenario)}<br>
            {html.escape(INTERVENTIONS[strategy]["note"])}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "*Illustrative values, not measured decibel changes. Real "
        "effectiveness would need field studies and validation."
    )


# ================================================================
# PAGE 7 — SYSTEM ARCHITECTURE
# ================================================================

elif page == "System Architecture":

    st.markdown(
        '<div class="section-tag">07 / SYSTEM DESIGN & PROJECT EVIDENCE</div>',
        unsafe_allow_html=True
    )
    st.header("System Architecture")

    st.write(
        "Review the project's data, model and presentation components."
    )

    file_status = {
        "Batch audio predictions": PREDICTION_FILE,
        "Event history": HISTORY_FILE,
        "Hotspot priority results": HOTSPOT_FILE,
        "Geographic map": MAP_FILE,
        "Trained sound classifier": MODEL_FILE,
        "Label encoder": ENCODER_FILE
    }

    status_rows = []

    for name, path in file_status.items():
        status_rows.append({
            "Component": name,
            "Status": "AVAILABLE" if path.exists() else "NOT FOUND",
            "Project file": str(path.relative_to(ROOT))
        })

    st.subheader("Project File Audit")

    st.dataframe(
        pd.DataFrame(status_rows),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Processing Pipeline")

    architecture = [
        ("01", "Audio Dataset", "UrbanSound8K and selected recordings"),
        ("02", "Feature Extraction", "40 MFCC and four spectral features"),
        ("03", "Machine Learning", "Saved sound-classification model"),
        ("04", "Event Analytics", "Rules applied to predicted sound classes"),
        ("05", "Spatial Analysis", "GIS map using assigned coordinates"),
        ("06", "Digital Twin", "Synthetic event and threshold simulation"),
        ("07", "Decision Support", "Intervention what-if comparison"),
        ("08", "Dashboard", "Interactive Streamlit interface")
    ]

    for number, name, description in architecture:
        st.markdown(f"""
        <div class="panel" style="padding:13px 17px;margin:8px 0;
             display:flex;gap:18px;align-items:center;min-height:65px;">
            <div style="min-width:35px;color:#4ae5dc;font-family:monospace;
                 font-size:19px;font-weight:800;">
                {number}
            </div>
            <div>
                <div style="font-size:14px;font-weight:800;color:#f0f8ff;">
                    {html.escape(name)}
                </div>
                <div style="font-size:12px;color:#9bb5c6;margin-top:5px;">
                    {html.escape(description)}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Project Limitations")

    st.markdown("""
    - The urban event simulator uses synthetic observations.
    - Actual recorded-audio predictions are displayed separately.
    - The assigned map coordinates are not live GPS measurements.
    - Priority scores based on predicted classes are not calibrated
      environmental noise levels.
    - Intervention effects are hypothetical assumptions.
    """)

    st.download_button(
        "Download project status report",
        data=pd.DataFrame(status_rows).to_csv(index=False).encode("utf-8"),
        file_name="auralis_project_file_status.csv",
        mime="text/csv"
    )


# ================================================================
# FOOTER
# ================================================================

st.divider()

st.markdown("""
<div style="text-align:center;padding:18px 8px;">
    <div style="font-size:20px;font-weight:800;letter-spacing:6px;
        color:#4ae5dc;">
        A U R A L I S
    </div>
    <div style="font-family:monospace;font-size:10px;letter-spacing:2px;
        color:#809eb2;margin-top:9px;">
        SMART URBAN NOISE MONITORING SYSTEM
    </div>
    <div style="font-size:11px;color:#6f8a9f;margin-top:8px;">
        Recorded audio intelligence • Interactive digital twin • GIS analytics
    </div>
</div>
""", unsafe_allow_html=True)