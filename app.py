"""
TFCM2-F Digital Twin Dashboard
Run with: streamlit run app.py
Requires: model_core.py + all .pkl files in the same directory
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="TFCM2-F Digital Twin",
    page_icon="⚡",
    initial_sidebar_state="collapsed"
)

# ── GLOBAL STYLES ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}
.stApp { background-color: #0d0f14; }

.panel-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #f97316;
    margin-bottom: 4px;
}
.main-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 600;
    color: #f1f5f9;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.sub-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 14px;
    color: #64748b;
    font-weight: 300;
    letter-spacing: 0.05em;
}
.param-card {
    background: #141720;
    border: 1px solid #1e2433;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.param-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.param-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 16px;
    color: #f97316;
    font-weight: 600;
}
div[data-testid="stPopover"] button {
    padding: 2px 8px !important;
    font-size: 10px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    color: #64748b !important;
    background: transparent !important;
    border: 1px solid #1e2433 !important;
    min-height: 0 !important;
    line-height: 1.4 !important;
    margin-top: -6px !important;
}
div[data-testid="stPopover"] button:hover {
    color: #f97316 !important;
    border-color: #f97316 !important;
}
.param-def-heading {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #f97316;
    margin-bottom: 6px;
}
.param-def-body {
    font-size: 12px;
    color: #cbd5e1;
    line-height: 1.5;
    margin-bottom: 8px;
}
.param-def-eq {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #94a3b8;
    background: #0d0f16;
    border: 1px solid #1e2433;
    border-radius: 4px;
    padding: 8px 10px;
}
.badge-active {
    display: inline-block;
    background: #f97316;
    color: #0d0f14;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 2px;
    letter-spacing: 0.1em;
}
.badge-predict {
    display: inline-block;
    background: #1e2433;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 2px;
    letter-spacing: 0.1em;
    border: 1px solid #2d3748;
}
.stButton > button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    border-radius: 4px;
}
.h-rule {
    border: none;
    border-top: 1px solid #1e2433;
    margin: 18px 0;
}
[data-testid="metric-container"] {
    background: #141720;
    border: 1px solid #1e2433;
    border-radius: 6px;
    padding: 12px 16px;
}
[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 20px;
    color: #f97316;
}

/* ── Panel 3 KPI cards ─────────────────────────────────────────────────── */
.kpi-card {
    background: #141720;
    border: 1px solid #1e2433;
    border-top-width: 2px;
    border-radius: 6px;
    padding: 10px 14px;
}
.kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 20px;
    color: #f97316;
    margin-top: 4px;
}

/* Remove default streamlit padding on top */
.block-container { padding-top: 1.5rem; }

/* ── Block editor ───────────────────────────────────────────────────────── */
.blk-row {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #141720;
    border: 1px solid #1e2433;
    border-radius: 6px;
    padding: 8px 10px;
    margin-bottom: 6px;
}
.blk-bar {
    width: 4px;
    align-self: stretch;
    border-radius: 2px;
    flex-shrink: 0;
}
.blk-step-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #475569;
    width: 20px;
    flex-shrink: 0;
    text-align: center;
}
.blk-preview {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #64748b;
    padding: 2px 2px 6px 34px;
    margin-top: -4px;
}

/* ── Model architecture narrative ───────────────────────────────────────── */
.narrative-step {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    color: #f97316;
    letter-spacing: 0.1em;
    margin-bottom: 2px;
}
.narrative-heading {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 4px;
}
.narrative-body {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 12px;
    color: #94a3b8;
    line-height: 1.6;
    margin-bottom: 2px;
}
.narrative-block {
    margin-bottom: 16px;
}
.narrative-closing {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 12px;
    color: #cbd5e1;
    line-height: 1.7;
    font-style: italic;
    border-top: 1px solid #1e2433;
    padding-top: 14px;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

# ── PLOTLY THEME ──────────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='#141720',
    plot_bgcolor='#0d0f14',
    font=dict(family='IBM Plex Mono', size=10, color='#94a3b8'),
    margin=dict(l=44, r=16, t=40, b=36),
    legend=dict(font=dict(size=8.5), bgcolor='rgba(13,15,20,0.65)',
                bordercolor='#1e2433', borderwidth=1,
                orientation='h', yanchor='top', y=0.99, xanchor='left', x=0.01),
    height=220,
    xaxis=dict(gridcolor='#1e2433', zerolinecolor='#1e2433', tickfont=dict(size=9)),
    yaxis=dict(gridcolor='#1e2433', zerolinecolor='#1e2433', tickfont=dict(size=9)),
)

COLORS = {
    'orange' : '#f97316',
    'blue'   : '#38bdf8',
    'green'  : '#4ade80',
    'purple' : '#a78bfa',
    'red'    : '#f87171',
    'teal'   : '#2dd4bf',
    'yellow' : '#fbbf24',
    'slate'  : '#64748b',
}

MEASURED_COLOR  = COLORS['blue']
PREDICTED_COLOR = COLORS['red']

# ── PLOT HELPERS & STEADY-STATE LOWESS FILTER ───────────────────────────────
def mini_plot(x, traces, title, y_label, height=220):
    fig = go.Figure()
    for y, name, color, dash in traces:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines', name=name,
            line=dict(color=color, width=1.4, dash=dash)))
    layout = {**PLOT_LAYOUT, 'height': height,
              'title': dict(text=title, font=dict(size=10, color='#94a3b8'), x=0),
              'yaxis': dict(gridcolor='#1e2433', zerolinecolor='#1e2433',
                            tickfont=dict(size=9),
                            title=dict(text=y_label, font=dict(size=9)))}
    fig.update_layout(**layout)
    return fig


def compute_lowess_polarization(current, voltage, dt, frac=0.3):
    """
    1. Filters out transient points (|dI/dt| > threshold).
    2. Fits a LOWESS (Locally Weighted Scatterplot Smoothing) regression 
       to extract the steady-state polarization curve.
    """
    di_dt = np.abs(np.gradient(current, dt))
    
    # Filter points where current is changing slowly (lower 65th percentile of rate-of-change)
    thresh = np.percentile(di_dt, 65)
    mask = di_dt <= thresh

    if np.sum(mask) < 10:
        mask = np.ones_like(current, dtype=bool)

    i_ss = current[mask]
    v_ss = voltage[mask]

    sort_idx = np.argsort(i_ss)
    i_ss_sorted = i_ss[sort_idx]
    v_ss_sorted = v_ss[sort_idx]

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        res = lowess(v_ss_sorted, i_ss_sorted, frac=frac, return_sorted=True)
        return res[:, 0], res[:, 1], i_ss, v_ss
    except ImportError:
        # Fallback: Local weighted Gaussian moving average if statsmodels isn't installed
        grid_i = np.linspace(i_ss_sorted.min(), i_ss_sorted.max(), 120)
        grid_v = []
        bandwidth = (i_ss_sorted.max() - i_ss_sorted.min()) * frac / 2.0
        bandwidth = max(bandwidth, 1.0)
        for gi in grid_i:
            weights = np.exp(-0.5 * ((i_ss_sorted - gi) / bandwidth) ** 2)
            if np.sum(weights) > 1e-5:
                grid_v.append(np.sum(weights * v_ss_sorted) / np.sum(weights))
            else:
                grid_v.append(np.nan)
        return grid_i, np.array(grid_v), i_ss, v_ss


# ── FUEL CELL HTML ────────────────────────────────────────────────────────────
def _fc_html(width=780, height=480, dim=False, grow=False):
    op = "0.55" if dim else "1.0"
    grow_css = """
      svg { animation: fcGrow 0.6s cubic-bezier(0.22,1,0.36,1) both; }
      @keyframes fcGrow {
        0%   { transform: scale(0.82); opacity: 0.0; }
        100% { transform: scale(1);    opacity: 1.0; }
      }
    """ if grow else ""

    return f"""<!DOCTYPE html>
<html><head>
<style>
  body {{ margin:0; padding:0; background:transparent;
          display:flex; justify-content:center; align-items:center;
          height:{height}px; overflow:hidden; }}
  {grow_css}
</style>
</head><body>
<svg width="{width}" height="{height}" viewBox="0 0 900 560"
     xmlns="http://www.w3.org/2000/svg" opacity="{op}">
  <defs>
    <linearGradient id="faceTop" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"  style="stop-color:#3a4358"/>
      <stop offset="100%" style="stop-color:#2b3346"/>
    </linearGradient>
    <linearGradient id="faceFront" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"  style="stop-color:#20263480"/>
      <stop offset="0%"  style="stop-color:#242b3a"/>
      <stop offset="100%" style="stop-color:#181d29"/>
    </linearGradient>
    <linearGradient id="faceSide" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"  style="stop-color:#181c27"/>
      <stop offset="100%" style="stop-color:#101319"/>
    </linearGradient>
    <pattern id="vent" width="7" height="7" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
      <rect width="7" height="7" fill="#12151d"/>
      <line x1="0" y1="0" x2="0" y2="7" stroke="#3a4358" stroke-width="2.2"/>
    </pattern>
    <radialGradient id="fanHub" cx="35%" cy="35%" r="70%">
      <stop offset="0%"  style="stop-color:#2d3748"/>
      <stop offset="100%" style="stop-color:#141720"/>
    </radialGradient>
    <marker id="aH2"  markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
      <path d="M0,0 L0,7 L9,3.5 z" fill="#4ade80"/></marker>
    <marker id="aAir" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
      <path d="M0,0 L0,7 L9,3.5 z" fill="#38bdf8"/></marker>
    <marker id="aW"   markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
      <path d="M0,0 L0,7 L9,3.5 z" fill="#a78bfa"/></marker>
    <marker id="aC"   markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">
      <path d="M0,0 L0,7 L9,3.5 z" fill="#f97316"/></marker>
  </defs>

  <ellipse cx="460" cy="472" rx="230" ry="16" fill="#000000" opacity="0.35"/>

  <polygon points="380,110 648,265 514,343 246,188" fill="url(#faceTop)"
           stroke="#475569" stroke-width="1.5"/>
  <polygon points="380,210 648,365 648,265 380,110" fill="url(#faceFront)"
           stroke="#475569" stroke-width="1.5"/>
  <polygon points="648,365 514,443 514,343 648,265" fill="url(#faceSide)"
           stroke="#475569" stroke-width="1.5"/>

  <line x1="380" y1="210" x2="648" y2="265" stroke="#3a4358" stroke-width="2"/>
  <line x1="380" y1="110" x2="648" y2="365" stroke="#3a4358" stroke-width="2"/>
  <line x1="380" y1="160" x2="648" y2="315" stroke="#3a4358" stroke-width="1.5" opacity="0.7"/>

  <polygon points="387,153 494,215 454,238 347,176" fill="url(#vent)" stroke="#475569" stroke-width="1"/>

  <g fill="#94a3b8" stroke="#0d0f14" stroke-width="0.5">
    <circle cx="380" cy="210" r="3.6"/><circle cx="648" cy="365" r="3.6"/>
    <circle cx="648" cy="265" r="3.6"/><circle cx="380" cy="110" r="3.6"/>
    <circle cx="514" cy="443" r="3.6"/><circle cx="514" cy="343" r="3.6"/>
    <circle cx="246" cy="288" r="3.6"/><circle cx="246" cy="188" r="3.6"/>
    <circle cx="469" cy="162" r="3"/><circle cx="559" cy="213" r="3"/>
    <circle cx="469" cy="262" r="3"/><circle cx="559" cy="313" r="3"/>
  </g>

  <rect x="428" y="235" width="14" height="10" rx="2" fill="#2d3748" stroke="#475569"/>
  <rect x="508" y="281" width="14" height="10" rx="2" fill="#2d3748" stroke="#475569"/>
  <rect x="588" y="328" width="14" height="10" rx="2" fill="#2d3748" stroke="#475569"/>

  <path d="M 610,290 C 690,278 750,310 828,292" fill="none"
        stroke="#f97316" stroke-width="11" stroke-linecap="round" opacity="0.92"/>
  <path d="M 610,290 C 690,278 750,310 828,292" fill="none"
        stroke="#fb923c" stroke-width="3" stroke-linecap="round" opacity="0.6"/>
  <ellipse cx="676" cy="281" rx="7" ry="10" fill="#1a1f2b" stroke="#0d0f14"/>
  <ellipse cx="758" cy="304" rx="7" ry="10" fill="#1a1f2b" stroke="#0d0f14"/>
  <text x="836" y="288" font-family="IBM Plex Mono" font-size="10" fill="#f97316" font-weight="600">HV+</text>
  <text x="836" y="302" font-family="IBM Plex Mono" font-size="10" fill="#f97316" font-weight="600">HV−</text>

  <circle cx="300" cy="303" r="38" fill="url(#fanHub)" stroke="#3a4358" stroke-width="2.5"/>
  <g transform="translate(300,303)">
    <animateTransform attributeName="transform" type="rotate"
      values="0 0 0;360 0 0" dur="1.8s" repeatCount="indefinite" additive="sum"/>
    <path d="M0,-29 Q13,-13 0,0 Q-13,-13 0,-29"  fill="#38bdf8" opacity="0.75"/>
    <path d="M29,0  Q13,13  0,0 Q13,-13  29,0"   fill="#38bdf8" opacity="0.75"/>
    <path d="M0,29  Q-13,13 0,0 Q13,13   0,29"   fill="#38bdf8" opacity="0.75"/>
    <path d="M-29,0 Q-13,-13 0,0 Q-13,13 -29,0"  fill="#38bdf8" opacity="0.75"/>
  </g>
  <circle cx="300" cy="303" r="7" fill="#0d0f14" stroke="#38bdf8" stroke-width="1.8"/>
  <text x="300" y="357" text-anchor="middle" font-family="IBM Plex Mono"
        font-size="9" fill="#64748b" letter-spacing="0.08em">COMPRESSOR</text>

  <rect x="418" y="132" width="98" height="22" rx="3" fill="#0d0f14" opacity="0.55"/>
  <text x="467" y="147" text-anchor="middle" font-family="IBM Plex Mono"
        font-size="11" fill="#94a3b8" font-weight="600" letter-spacing="0.1em">330 CELLS</text>

  <line x1="55" y1="150" x2="332" y2="195" stroke="#4ade80" stroke-width="3"
        marker-end="url(#aH2)" stroke-dasharray="7 4">
    <animate attributeName="stroke-dashoffset" values="22;0" dur="0.8s" repeatCount="indefinite"/>
  </line>
  <text x="30" y="132" font-family="IBM Plex Mono" font-size="13" fill="#4ade80" font-weight="600">H₂</text>
  <text x="14" y="148" font-family="IBM Plex Mono" font-size="9" fill="#4ade80" letter-spacing="0.06em">INLET</text>

  <line x1="55" y1="368" x2="278" y2="318" stroke="#38bdf8" stroke-width="2.6"
        marker-end="url(#aAir)" stroke-dasharray="7 4">
    <animate attributeName="stroke-dashoffset" values="22;0" dur="1.1s" repeatCount="indefinite"/>
  </line>
  <text x="14" y="396" font-family="IBM Plex Mono" font-size="13" fill="#38bdf8" font-weight="600">AIR</text>
  <text x="4"  y="411" font-family="IBM Plex Mono" font-size="9" fill="#38bdf8" letter-spacing="0.06em">INTAKE</text>

  <line x1="298" y1="510" x2="392" y2="225" stroke="#f97316" stroke-width="3"
        marker-end="url(#aC)" stroke-dasharray="7 4">
    <animate attributeName="stroke-dashoffset" values="22;0" dur="1.3s" repeatCount="indefinite"/>
  </line>
  <text x="228" y="528" font-family="IBM Plex Mono" font-size="10" fill="#f97316" font-weight="600">COOLANT IN</text>

  <line x1="602" y1="380" x2="710" y2="478" stroke="#f97316" stroke-width="3"
        marker-end="url(#aC)" stroke-dasharray="7 4" opacity="0.65">
    <animate attributeName="stroke-dashoffset" values="0;22" dur="1.3s" repeatCount="indefinite"/>
  </line>
  <text x="656" y="500" font-family="IBM Plex Mono" font-size="10" fill="#f97316" font-weight="600">COOLANT OUT</text>

  <line x1="600" y1="252" x2="770" y2="132" stroke="#a78bfa" stroke-width="2.6"
        marker-end="url(#aW)" stroke-dasharray="6 4">
    <animate attributeName="stroke-dashoffset" values="0;20" dur="1.6s" repeatCount="indefinite"/>
  </line>
  <text x="774" y="126" font-family="IBM Plex Mono" font-size="12" fill="#a78bfa" font-weight="600">H₂O</text>
  <text x="768" y="142" font-family="IBM Plex Mono" font-size="9" fill="#a78bfa" letter-spacing="0.06em">EXHAUST</text>

  <text x="460" y="55" text-anchor="middle" font-family="IBM Plex Mono"
        font-size="18" fill="#f1f5f9" font-weight="600"
        letter-spacing="0.16em">TFCM2-F</text>

  <text x="460" y="547" text-anchor="middle" font-family="IBM Plex Mono"
        font-size="9" fill="#475569" letter-spacing="0.06em">
    1270 × 630 × 410 mm  |  Toyota Fuel Cell Stack Module
  </text>
</svg>
</body></html>"""


# ── SESSION STATE ─────────────────────────────────────────────────────────────
_defaults = {
    'panel'        : 1,
    'df_input'     : None,
    'dt'           : 0.1,
    't_start_K'    : 295.15,
    'results'      : None,
    'has_real_data': False,
    'input_mode'   : 'csv',
    'models'       : None,
    'model_error'  : None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── MODEL LOADER ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    from model_core import load_all_models
    return load_all_models('.')

def ensure_models():
    if st.session_state.models is None and st.session_state.model_error is None:
        try:
            st.session_state.models = load_models()
        except Exception as e:
            st.session_state.model_error = str(e)

ensure_models()


# ==============================================================================
# PANEL 1 — INPUT CONFIGURATION
# ==============================================================================
@st.dialog("Model Architecture", width="large")
def _show_architecture_dialog():
    _narrative = [
        ("01 · THE REQUEST ARRIVES", "The Request Arrives",
         "Every simulation starts as one number changing over time: power "
         "request. That's all a drive cycle really is, a target the stack "
         "has to chase. But a real TFCM2-F doesn't just jump to that number. "
         "Dozens of physical quantities have to shift underneath it first, "
         "current density, gas pressures, coolant flow, and none of those "
         "are known yet. They have to be inferred."),
        ("02 · GUESSING THE SENSORS", "Guessing the Sensors",
         "The first thing the twin does is stand in for six sensors it "
         "doesn't have. Six lightweight models, mostly Random Forests plus "
         "one linear model for H₂ pressure, look at the power request and "
         "how fast it's changing, and predict what current density, air "
         "pressure, bypass flow, and coolant pump speed would be at that "
         "operating point. Same relationships a real ECU would be reading "
         "off physical sensors on the actual stack."),
        ("03 · WAITING FOR THE COOLANT", "Waiting for the Coolant to Catch Up",
         "Temperature doesn't move as fast as power does. A nonlinear "
         "state-space integrator tracks coolant inlet temperature second "
         "by second, balancing heat gained from the load against cooling "
         "supplied by the pump and bypass valve. This is what lets the "
         "twin know how warm or cold the stack's inlet actually is at each "
         "timestep, not just what it would eventually settle to."),
        ("04 · THE PHYSICS CORE", "The Physics Core Does the Real Work",
         "With current density, pressures, and inlet temperature "
         "resolved, the physics core takes over. This is where the actual "
         "electrochemistry lives. Nernst voltage sets the theoretical "
         "ceiling, activation and ohmic losses pull it down based on "
         "membrane hydration (Springer's model) and temperature "
         "(Arrhenius-corrected exchange current), and mass transport "
         "losses bite hardest at high load. The loop also tracks the "
         "stack's own thermal mass, so voltage and temperature evolve "
         "together rather than separately."),
        ("05 · WHAT PHYSICS MISSES", "The Physics Model Admits What It Doesn't Know",
         "No physics model captures everything, manufacturing "
         "tolerances, degradation, small effects nobody bothered to "
         "derive by hand. Rather than pretend the physics loop is "
         "perfect, an SVR trained on the gap between physics predictions "
         "and real measured voltage learns that residual and corrects "
         "for it. The physics core gets you most of the way there, the "
         "SVR closes the last few percent."),
        ("06 · POWER THE STACK KEEPS", "Turning Voltage Into Power the Stack Actually Keeps",
         "Final voltage times current gives gross power, but the stack "
         "doesn't keep all of it. The air compressor follows a fan "
         "affinity law, cubic with speed, and the coolant pump, H₂ pump, "
         "and DC-DC converter each take their own cut. What's left after "
         "all three is net power, the number that actually matters to "
         "whoever's driving."),
        ("07 · COUNTING THE HYDROGEN", "Counting the Hydrogen",
         "Faraday's law gives a hard physical floor on hydrogen "
         "consumption from current draw alone, but real consumption runs "
         "slightly above that floor. A small ML residual, trained on the "
         "gap, adds the difference back, so the estimate tracks real "
         "measured usage instead of just the theoretical minimum."),
    ]
    for step_label, heading, body in _narrative:
        st.markdown(f"""
        <div class="narrative-block">
          <div class="narrative-step">{step_label}</div>
          <div class="narrative-heading">{heading}</div>
          <div class="narrative-body">{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        '<div class="narrative-closing">'
        "This is the value proposition of a validated digital twin: it "
        "allows a stack to be evaluated across operating conditions that "
        "would be costly, time consuming, or impractical to test on the "
        "physical hardware directly, while remaining anchored to measured "
        "data through the SVR correction and parameter identification "
        "described above. The same modeling approach extends naturally to "
        "medium and heavy duty vehicle drive cycles, and to marine "
        "propulsion profiles, where hydrogen fuel cells are increasingly "
        "studied as an alternative to conventional combustion. As more "
        "measured data becomes available, digital twins of this kind offer "
        "a practical path for testing that scenario before committing to "
        "the cost of physical trials."
        '</div>', unsafe_allow_html=True)


def render_panel1():
    st.markdown('<div class="panel-title">Toyota KAUST Clean Combustion Research Center</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="main-title">TFCM2-F Digital Twin</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Physics-ML Hybrid Stack Simulation</div>',
                unsafe_allow_html=True)

    if st.session_state.model_error:
        st.error(
            f"**Model files not found.** Run your notebook first to generate all .pkl files.\n\n"
            f"`{st.session_state.model_error}`")
        return

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)

    left, mid, right = st.columns([1.1, 1.5, 1.0], gap="large")

    with left:
        st.markdown('<div class="panel-title">Ambient Conditions</div>', unsafe_allow_html=True)

        t_room = st.slider("Starting Room Temperature (°C)", 15.0, 60.0, 22.0, 0.5)
        rh_val = st.slider("Relative Humidity", 0.0, 1.0, 0.48, 0.01, format="%.2f")
        st.session_state.t_start_K = t_room + 273.15

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        st.caption(
            "Sets the stack's initial thermal and humidity state before the "
            "drive cycle begins. These feed directly into the physics core's "
            "Springer membrane hydration model and the T_in thermal integrator.")

    with mid:
        st.markdown('<div class="panel-title">Drive Cycle Input</div>', unsafe_allow_html=True)

        mode = st.radio("Drive cycle input mode", ["Upload CSV", "Drive Cycle Builder"],
                        horizontal=True, label_visibility="collapsed")

        if mode == "Upload CSV":
            st.session_state.input_mode = 'csv'
            f = st.file_uploader(
                "VehicleSpy / FC_A export, or a simple time + power CSV",
                type=["csv"], label_visibility="collapsed")
            if f:
                try:
                    from model_core import process_fc_data_from_upload_auto
                    df, dt, t_s, is_simple = process_fc_data_from_upload_auto(f)
                    st.session_state.df_input      = df
                    st.session_state.dt            = dt
                    # A simple power+time file carries no sensor channels
                    # beyond power and time, so there's nothing measured to
                    # overlay against, same as a function-block-built cycle.
                    st.session_state.has_real_data = not is_simple

                    if is_simple:
                        st.info(
                            f"Loaded {len(df):,} rows  |  dt={dt:.3f}s  |  "
                            f"simple power+time CSV detected, no sensor "
                            f"channels found. ECUs will predict every other "
                            f"channel from power alone (PREDICT MODE). "
                            f"Starting temperature is taken from the slider "
                            f"above, not the file.")
                    else:
                        st.success(
                            f"Loaded {len(df):,} rows  |  dt={dt:.3f}s  |  "
                            f"T_start={t_s-273.15:.1f}°C")

                    fig = go.Figure(go.Scatter(
                        x=df['time'], y=df['power_request']/1000,
                        mode='lines', line=dict(color=COLORS['orange'], width=1.2)))
                    fig.update_layout(**{**PLOT_LAYOUT, 'height':160,
                                        'title':dict(text='Power Request (kW)', font=dict(size=10))})
                    st.plotly_chart(fig, width="stretch")
                except Exception as e:
                    st.error(f"CSV parse error: {e}")
                    st.session_state.df_input      = None
                    st.session_state.has_real_data = False
            else:
                st.session_state.df_input      = None
                st.session_state.has_real_data = False

        else:
            st.session_state.input_mode    = 'blocks'
            st.session_state.has_real_data = False
            render_block_editor()

    with right:
        st.markdown('<div class="panel-title">Model Info</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:12px;color:#64748b;line-height:1.5;margin-top:2px">'
            'Physics-ML hybrid twin combining a convective thermal engine, '
            'six ECU sub-models, and an SVR residual correction layer.</p>',
            unsafe_allow_html=True)

        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        if st.button("📖  Learn more about the model architecture",
                    width="stretch"):
            _show_architecture_dialog()

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)

    components.html(_fc_html(width=820, height=440), height=444, scrolling=False)

    ready = (st.session_state.df_input is not None
             and st.session_state.models is not None)

    btn_l, btn_c, btn_r = st.columns([1.4, 1, 1.4])
    with btn_c:
        if ready:
            n   = len(st.session_state.df_input)
            dur = n * st.session_state.dt
            if st.session_state.has_real_data:
                st.markdown(
                    '<div style="text-align:center"><div class="badge-active">'
                    'READY  ·  CSV MODE</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="text-align:center"><div class="badge-predict">'
                    'READY  ·  PREDICT MODE</div></div>', unsafe_allow_html=True)
            st.markdown(
                f'<p style="text-align:center;font-size:11px;color:#64748b;'
                f'font-family:\'IBM Plex Mono\',monospace;margin:6px 0">'
                f'{n:,} steps · {dur:.0f}s · dt={st.session_state.dt:.3f}s</p>',
                unsafe_allow_html=True)

        if st.button(
            "▶  RUN DIGITAL TWIN" if ready else "⚠  Load a drive cycle first",
            width="stretch",
            type="primary"   if ready else "secondary",
            disabled=not ready,
        ):
            st.session_state.panel = 2
            st.rerun()


# ==============================================================================
# BLOCK EDITOR — Vehicle Spy style drive cycle builder
# ==============================================================================

BLOCK_LABELS = {'set': 'SET', 'wait': 'WAIT', 'ramp': 'RAMP'}
BLOCK_COLORS = {'set': COLORS['orange'], 'wait': COLORS['blue'], 'ramp': COLORS['purple']}

_block_defaults = {
    'cycle_blocks': [],
    'next_block_id': 0,
}
for k, v in _block_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _resolve_blocks_to_steps(blocks):
    steps = []
    state = 0.0
    for b in blocks:
        if b['block_type'] == 'set':
            state = b['power_kw']
        elif b['block_type'] == 'wait':
            steps.append({
                'type': 'constant',
                'power_kw': state,
                'power_start_kw': state,
                'duration_s': b['duration_s'],
            })
        elif b['block_type'] == 'ramp':
            steps.append({
                'type': 'ramp',
                'power_kw': b['power_kw'],
                'power_start_kw': state,
                'duration_s': b['duration_s'],
            })
            state = b['power_kw']
    return steps


def _block_preview_text(block, running_state, running_time):
    if block['block_type'] == 'set':
        return f"→ jumps to {block['power_kw']:.0f} kW instantly (at t={running_time:.0f}s)"
    if block['block_type'] == 'wait':
        return (f"→ holds {running_state:.0f} kW for {block['duration_s']}s "
                f"(t={running_time:.0f}s → {running_time + block['duration_s']:.0f}s)")
    if block['block_type'] == 'ramp':
        return (f"→ ramps {running_state:.0f} kW → {block['power_kw']:.0f} kW "
                f"over {block['duration_s']}s "
                f"(t={running_time:.0f}s → {running_time + block['duration_s']:.0f}s)")
    return ""


def render_block_editor():
    blocks = st.session_state.cycle_blocks

    cap_l, cap_r = st.columns([3, 1])
    with cap_l:
        st.caption(
            "Stack blocks top to bottom, like a Vehicle Spy script. **Set Value** "
            "jumps instantly, **Wait For** holds, **Ramp To** transitions from "
            "whatever the value currently is.")
    with cap_r:
        if st.button("🗑 Clear All", width="stretch",
                    disabled=(len(blocks) == 0)):
            st.session_state.cycle_blocks = []
            st.session_state.df_input     = None
            st.rerun()

    if not blocks:
        st.info("No blocks yet. Start with **Set Value** below to set an "
                 "initial power, then add **Wait For** and **Ramp To** blocks "
                 "to build out the cycle.")

    running_state = 0.0
    running_time = 0.0
    for idx, b in enumerate(blocks):
        bid = b['id']
        color = BLOCK_COLORS[b['block_type']]

        c_bar, c_num, c_type, c_fields, c_up, c_down, c_del = st.columns(
            [0.06, 0.22, 1.0, 2.45, 0.3, 0.3, 0.3])

        with c_bar:
            st.markdown(
                f'<div style="width:4px;height:32px;background:{color};'
                f'border-radius:2px;margin-top:24px;"></div>',
                unsafe_allow_html=True)
        with c_num:
            st.markdown(
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;'
                f'color:#475569;margin-top:30px;">{idx+1}</div>',
                unsafe_allow_html=True)
        with c_type:
            new_type = st.selectbox(
                "Type", ['set', 'wait', 'ramp'],
                index=['set', 'wait', 'ramp'].index(b['block_type']),
                format_func=lambda x: BLOCK_LABELS[x],
                key=f"btype_{bid}")
            b['block_type'] = new_type

        with c_fields:
            if b['block_type'] == 'set':
                b['power_kw'] = st.number_input(
                    "Power (kW)", value=float(b.get('power_kw', 30.0)),
                    min_value=0.0, max_value=120.0, step=5.0,
                    key=f"pw_{bid}")
            elif b['block_type'] == 'wait':
                b['duration_s'] = st.number_input(
                    "Time (s)", value=int(b.get('duration_s', 60)),
                    min_value=1, max_value=3600, step=10,
                    key=f"dur_{bid}")
            else:  # ramp
                rc1, rc2 = st.columns(2)
                with rc1:
                    b['power_kw'] = st.number_input(
                        "To Power (kW)", value=float(b.get('power_kw', 50.0)),
                        min_value=0.0, max_value=120.0, step=5.0,
                        key=f"rpw_{bid}")
                with rc2:
                    b['duration_s'] = st.number_input(
                        "Time (s)", value=int(b.get('duration_s', 30)),
                        min_value=1, max_value=3600, step=5,
                        key=f"rdur_{bid}")

        with c_up:
            st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
            if st.button("↑", key=f"up_{bid}", disabled=(idx == 0),
                        width="stretch"):
                blocks[idx - 1], blocks[idx] = blocks[idx], blocks[idx - 1]
                st.rerun()
        with c_down:
            st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
            if st.button("↓", key=f"down_{bid}", disabled=(idx == len(blocks) - 1),
                        width="stretch"):
                blocks[idx + 1], blocks[idx] = blocks[idx], blocks[idx + 1]
                st.rerun()
        with c_del:
            st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
            if st.button("✕", key=f"del_{bid}", width="stretch"):
                blocks.pop(idx)
                st.rerun()

        st.markdown(
            f'<div class="blk-preview">{_block_preview_text(b, running_state, running_time)}</div>',
            unsafe_allow_html=True)

        if b['block_type'] in ('set', 'ramp'):
            running_state = b['power_kw']
        if b['block_type'] in ('wait', 'ramp'):
            running_time += b['duration_s']

    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("＋ Set Value", width="stretch"):
            last = running_state
            st.session_state.cycle_blocks.append({
                'id': st.session_state.next_block_id,
                'block_type': 'set', 'power_kw': last})
            st.session_state.next_block_id += 1
            st.rerun()
    with a2:
        if st.button("＋ Wait For", width="stretch"):
            st.session_state.cycle_blocks.append({
                'id': st.session_state.next_block_id,
                'block_type': 'wait', 'duration_s': 60})
            st.session_state.next_block_id += 1
            st.rerun()
    with a3:
        if st.button("＋ Ramp To", width="stretch"):
            last = running_state
            st.session_state.cycle_blocks.append({
                'id': st.session_state.next_block_id,
                'block_type': 'ramp', 'power_kw': last + 20.0, 'duration_s': 30})
            st.session_state.next_block_id += 1
            st.rerun()

    if blocks:
        from model_core import build_drive_cycle_from_table
        steps = _resolve_blocks_to_steps(blocks)
        if steps:
            df, dt = build_drive_cycle_from_table(steps)
            st.session_state.df_input = df
            st.session_state.dt       = dt
            total_s = sum(s['duration_s'] for s in steps)
            fig = go.Figure(go.Scatter(
                x=df['time'], y=df['power_request']/1000,
                mode='lines', line=dict(color=COLORS['orange'], width=1.5)))
            fig.update_layout(**{**PLOT_LAYOUT, 'height':140,
                                 'title':dict(text='Cycle Preview (kW)',
                                              font=dict(size=10))})
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Duration: {total_s}s  |  {len(df):,} timesteps @ 0.1s")
        else:
            st.session_state.df_input = None
            st.info("Add a Wait For or Ramp To block to generate timesteps "
                     "(Set Value alone has zero duration).")


# ==============================================================================
# PANEL 2 — LOADING / INFERENCE
# ==============================================================================
def render_panel2():
    st.markdown("""
    <style>
      [data-testid="stHeader"] {
          opacity: 0; max-height: 0; overflow: hidden;
          transition: opacity 0.25s ease, max-height 0.25s ease;
          pointer-events: none;
      }
      .block-container {
          display: flex;
          flex-direction: column;
          justify-content: flex-start;
          align-items: center;
          min-height: 100vh;
          padding-top: 4vh;
          padding-bottom: 4vh;
          overflow-y: auto;
      }
      .block-container > div { width: 100%; }
      @keyframes fcFadeIn {
        0%   { opacity: 0; transform: translateY(4px); }
        100% { opacity: 1; transform: translateY(0); }
      }
      .stage-label { animation: fcFadeIn 0.3s ease both; }
    </style>
    <script>
      window.parent.scrollTo({top: 0, behavior: 'instant'});
      const _scrollLock = setInterval(() => {
        window.parent.scrollTo({top: 0, behavior: 'instant'});
      }, 300);
      setTimeout(() => clearInterval(_scrollLock), 120000);
    </script>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([0.6, 2.8, 0.6])
    with col:
        st.markdown(
            '<div class="panel-title" style="text-align:center;letter-spacing:0.24em">'
            'DIGITAL TWIN EXECUTING</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="main-title" style="text-align:center;font-size:24px;'
            'margin-bottom:2px">TFCM2-F</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-title" style="text-align:center">'
            'Physics-ML inference pipeline running</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

        components.html(_fc_html(width=760, height=430, dim=False, grow=True),
                        height=434, scrolling=False)

        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

        stages = [
            ("ECU PIPELINE",    "Current density, pressures, flow rates"),
            ("T_in INTEGRATOR", "Nonlinear actuator thermal state-space"),
            ("PHYSICS LOOP",    "Convective Arrhenius voltage integration"),
            ("SVR CORRECTION",  "Residual ML correction layer"),
            ("AUX & H₂",        "Auxiliary power, hydrogen consumption"),
        ]

        prog_bar  = st.progress(0)
        status_ph = st.empty()

        try:
            from model_core import run_inference

            steps_per_stage = 8
            total_steps     = len(stages) * steps_per_stage
            step_count      = 0

            for i, (stage, detail) in enumerate(stages):
                status_ph.markdown(
                    f'<p class="stage-label" style="font-family:\'IBM Plex Mono\',monospace;'
                    f'font-size:11px;color:#64748b;text-align:center;letter-spacing:0.1em;'
                    f'margin:4px 0">{stage} · {detail}</p>',
                    unsafe_allow_html=True)

                is_last_stage = (i == len(stages) - 1)

                for s in range(steps_per_stage):
                    step_count += 1
                    prog_bar.progress(step_count / total_steps)

                    if is_last_stage and s == steps_per_stage - 1:
                        n_steps = len(st.session_state.df_input)
                        status_ph.markdown(
                            f'<p class="stage-label" style="font-family:\'IBM Plex Mono\','
                            f'monospace;font-size:11px;color:#64748b;text-align:center;'
                            f'letter-spacing:0.1em;margin:4px 0">RUNNING PHYSICS LOOP · '
                            f'{n_steps:,} timesteps — may take longer for larger cycles</p>',
                            unsafe_allow_html=True)
                        with st.spinner(""):
                            results = run_inference(
                                st.session_state.df_input,
                                st.session_state.dt,
                                st.session_state.t_start_K,
                                st.session_state.models,
                            )
                    else:
                        time.sleep(0.45 / steps_per_stage)

            st.session_state.results = results
            prog_bar.progress(1.0)
            status_ph.markdown(
                '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;'
                'color:#4ade80;text-align:center;letter-spacing:0.18em;margin:6px 0">'
                '✓  COMPLETE</p>',
                unsafe_allow_html=True)
            time.sleep(0.55)
            st.session_state.panel = 3
            st.rerun()

        except Exception as e:
            st.error(f"Inference error: {e}")
            if st.button("← Back to Input"):
                st.session_state.panel = 1
                st.rerun()


# ==============================================================================
# PANEL 3 — TELEMETRY DASHBOARD
# ==============================================================================
def render_panel3():
    r = st.session_state.results
    if r is None:
        st.session_state.panel = 1
        st.rerun()

    t        = r['time']
    has_real = st.session_state.has_real_data

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    hl, hr = st.columns([3, 1])
    with hl:
        st.markdown(
            '<div class="panel-title">Toyota KAUST CCRC  ·  TFCM2-F Digital Twin</div>',
            unsafe_allow_html=True)
        st.markdown(
            '<div class="main-title" style="font-size:20px">Telemetry Dashboard</div>',
            unsafe_allow_html=True)
        if has_real:
            st.markdown(
                '<div class="sub-title">📊  Measured vs Predicted — CSV validation mode</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="sub-title">🔮  Prediction-only mode — twin outputs</div>',
                unsafe_allow_html=True)

    with hr:
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        if st.button("← New Simulation", width="stretch"):
            st.session_state.panel         = 1
            st.session_state.results       = None
            st.session_state.has_real_data = False
            st.rerun()
        
        out_cols = {
            'time_s'             : t,
            'power_request_kW'   : r['power_request'],
            'current_density_Am2': r['current_density'],
            'voltage_pred_V'     : r['V_final'],
            'T_in_pred_C'        : r['T_in_K']    - 273.15,
            'T_out_pred_C'       : r['T_out_K']   - 273.15,
            'T_stack_C'          : r['T_stack_K'] - 273.15,
            'h2_pressure_Pa'     : r['h2_p'],
            'air_pressure_Pa'    : r['air_p'],
            'bypass_flow_kgs'    : r['bypass_flow'],
            'wp_rps'             : r['wp_rps'],
            'power_gross_kW'     : r['p_gross_kw'],
            'power_net_kW'       : r['p_net_kw'],
            'power_aux_kW'       : r['p_aux_kw'],
            'h2_rate_mgs'        : r['h2_rate_mgs'],
            'h2_cumulative_g'    : r['h2_cumulative_g'],
        }
        if has_real:
            out_cols['voltage_measured_V'] = r['V_real']
            out_cols['T_in_measured_C']    = r['T_in_real']  - 273.15
            out_cols['T_out_measured_C']   = r['T_out_real'] - 273.15
        st.download_button(
            "⬇ Download CSV",
            data=pd.DataFrame(out_cols).to_csv(index=False).encode(),
            file_name="tfcm2f_twin_results.csv",
            mime="text/csv",
            width="stretch")

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────────────────────
    kpi_border = COLORS['orange'] if has_real else COLORS['slate']

    if has_real:
        ss_res = np.sum((r['V_real'] - r['V_final']) ** 2)
        ss_tot = np.sum((r['V_real'] - np.mean(r['V_real'])) ** 2)
        polarization_r2 = f"{(1 - ss_res / ss_tot):.4f}" if ss_tot > 0 else "N/A"
    else:
        polarization_r2 = "N/A"

    kpi_items = [
        ("Peak Voltage",   f"{r['V_final'].max():.1f} V"),
        ("Polarization R²", polarization_r2),
        ("Peak Net Power", f"{r['p_net_kw'].max():.1f} kW"),
        ("Total H₂",       f"{r['h2_cumulative_g'][-1]:.1f} g"),
        ("Peak T_stack",   f"{r['T_stack_K'].max()-273.15:.1f} °C"),
        ("Net/Gross Ratio",
         f"{(r['p_net_kw']/np.maximum(r['p_gross_kw'],0.001)).mean()*100:.1f} %"),
        ("Cycle Duration", f"{t[-1]:.0f} s"),
    ]
    kpi_cols = st.columns(7)
    for col, (label, val) in zip(kpi_cols, kpi_items):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-top-color:{kpi_border};">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    def overlay(pred_y, pred_label, real_y=None, real_label="Measured"):
        if has_real and real_y is not None:
            return [
                (real_y,  real_label,  MEASURED_COLOR,  'solid'),
                (pred_y,  pred_label,  PREDICTED_COLOR, 'dash'),
            ]
        return [(pred_y, pred_label, PREDICTED_COLOR, 'solid')]

    st.markdown('<div class="panel-title" style="margin-top:4px">Key Metrics</div>',
                unsafe_allow_html=True)
    hp1, hp2, hp3 = st.columns(3)

    with hp1:
        fig = go.Figure()
        dt_val = st.session_state.dt

        if has_real:
            x_m_curve, y_m_curve, x_m_ss, y_m_ss = compute_lowess_polarization(
                r['current_density_real'], r['V_real'], dt_val
            )
            # Faint background scatter (hidden from legend)
            fig.add_trace(go.Scatter(
                x=x_m_ss, y=y_m_ss, mode='markers', 
                marker=dict(color=MEASURED_COLOR, size=3, opacity=0.22),
                showlegend=False
            ))
            # Smooth LOWESS trend line
            fig.add_trace(go.Scatter(
                x=x_m_curve, y=y_m_curve, mode='lines', name='Measured',
                line=dict(color=MEASURED_COLOR, width=2.2)
            ))

        x_t_curve, y_t_curve, x_t_ss, y_t_ss = compute_lowess_polarization(
            r['current_density'], r['V_final'], dt_val
        )
        # Faint background scatter (hidden from legend)
        fig.add_trace(go.Scatter(
            x=x_t_ss, y=y_t_ss, mode='markers', 
            marker=dict(color=PREDICTED_COLOR, size=3, opacity=0.22),
            showlegend=False
        ))
        # Smooth LOWESS trend line
        fig.add_trace(go.Scatter(
            x=x_t_curve, y=y_t_curve, mode='lines', name='Twin',
            line=dict(color=PREDICTED_COLOR, width=2.2)
        ))

        fig.update_layout(**{**PLOT_LAYOUT,
            # Compact legend tucked into the top right
            'legend': dict(yanchor="top", y=0.99, xanchor="right", x=0.99, 
                           bgcolor='rgba(13,15,20,0.85)', bordercolor='#1e2433', 
                           borderwidth=1, font=dict(size=9)),
            'title': dict(text='Steady-State Polarization Curve (LOWESS)', font=dict(size=10)),
            'xaxis': dict(title='Current Density (A/m²)',
                          gridcolor='#1e2433', tickfont=dict(size=9)),
            'yaxis': dict(title='Stack Voltage (V)',
                          gridcolor='#1e2433', tickfont=dict(size=9))})
        st.plotly_chart(fig, width="stretch")

    with hp2:
        st.plotly_chart(mini_plot(t,
            overlay(r['h2_rate_mgs'], 'Twin H₂ Rate',
                    r['h2_rate_real_mgs'], 'Measured'),
            'H₂ Consumption Rate (mg/s)', 'mg/s'), width="stretch")

    with hp3:
        st.plotly_chart(mini_plot(t,
            overlay(r['p_net_kw'], 'Twin Net Power',
                    r['p_net_real_kw'], 'Measured'),
            'Net Power Output (kW)', 'kW'), width="stretch")

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    with st.expander("🌡  Thermal Tracking"):
        tc1, tc2 = st.columns(2)
        with tc1:
            st.plotly_chart(mini_plot(t,
                overlay(r['T_in_K']-273.15,   'T_in Twin',
                        r['T_in_real']-273.15, 'T_in Measured'),
                'Coolant Inlet T_in (°C)', '°C'), width="stretch")
            st.plotly_chart(mini_plot(t,
                [(r['T_stack_K']-273.15, 'T_stack', COLORS['yellow'], 'solid')],
                'Stack Core Temp T_stack (°C)', '°C'), width="stretch")
        with tc2:
            st.plotly_chart(mini_plot(t,
                overlay(r['T_out_K']-273.15,   'T_out Twin',
                        r['T_out_real']-273.15, 'T_out Measured'),
                'Coolant Outlet T_out (°C)', '°C'), width="stretch")

            fig = go.Figure()
            if has_real:
                fig.add_trace(go.Scatter(x=t, y=r['bypass_flow_real']*1000, mode='lines',
                                         name='Bypass Measured',
                                         line=dict(color=MEASURED_COLOR, width=1.4, dash='solid'),
                                         yaxis='y'))
                fig.add_trace(go.Scatter(x=t, y=r['wp_rps_real'], mode='lines',
                                         name='Pump Measured',
                                         line=dict(color=MEASURED_COLOR, width=1.4, dash='dot'),
                                         yaxis='y2'))
            fig.add_trace(go.Scatter(x=t, y=r['bypass_flow']*1000, mode='lines',
                                     name='Bypass Twin',
                                     line=dict(color=PREDICTED_COLOR, width=1.4, dash='solid'),
                                     yaxis='y'))
            fig.add_trace(go.Scatter(x=t, y=r['wp_rps'], mode='lines',
                                     name='Pump Twin',
                                     line=dict(color=PREDICTED_COLOR, width=1.4, dash='dot'),
                                     yaxis='y2'))
            fig.update_layout(**{**PLOT_LAYOUT,
                'title': dict(text='Coolant Flow Actuators', font=dict(size=10)),
                'yaxis': dict(title='Bypass Flow (g/s)', gridcolor='#1e2433',
                              tickfont=dict(size=9)),
                'yaxis2': dict(title='Pump Speed (rps)', overlaying='y', side='right',
                              gridcolor='#1e2433', tickfont=dict(size=9),
                              showgrid=False)})
            st.plotly_chart(fig, width="stretch")

    with st.expander("⛽  Gas Monitoring"):
        gc1, gc2 = st.columns(2)
        with gc1:
            st.plotly_chart(mini_plot(t,
                overlay(r['h2_p']/1000, 'Twin H₂ Pressure',
                        r['h2_p_real']/1000, 'Measured'),
                'H₂ Intake Pressure (kPa)', 'kPa'), width="stretch")
        with gc2:
            st.plotly_chart(mini_plot(t,
                overlay(r['air_p']/1000, 'Twin Air Pressure',
                        r['air_p_real']/1000, 'Measured'),
                'Air Intake Pressure (kPa)', 'kPa'), width="stretch")

    with st.expander("⚡  Electrical Outputs"):
        ec1, ec2 = st.columns(2)
        with ec1:
            st.plotly_chart(mini_plot(t,
                overlay(r['current_density'], 'Twin Current',
                        r['current_density_real'], 'Measured'),
                'Output Current Density (A/m²)', 'A/m²'), width="stretch")
        with ec2:
            st.plotly_chart(mini_plot(t,
                overlay(r['V_final'], 'Twin Voltage',
                        r['V_real'],  'Measured'),
                'Stack Voltage (V)', 'V'), width="stretch")

    with st.expander("🔧  Auxiliary Breakdown"):
        ac1, ac2 = st.columns(2)
        with ac1:
            st.plotly_chart(mini_plot(t,
                overlay(r['p_gross_kw'], 'Twin Gross Power',
                        r['p_gross_real_kw'], 'Measured'),
                'Gross Power Output (kW)', 'kW'), width="stretch")
        with ac2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=r['p_comp_kw'],      mode='lines',
                                     name='Compressor',
                                     line=dict(color=COLORS['yellow'], width=1.2),
                                     stackgroup='aux'))
            fig.add_trace(go.Scatter(x=t, y=r['p_aux_small_kw'], mode='lines',
                                     name='Pumps',
                                     line=dict(color=COLORS['purple'], width=1.2),
                                     stackgroup='aux'))
            fig.add_trace(go.Scatter(x=t, y=r['p_converter_kw'], mode='lines',
                                     name='DC-DC',
                                     line=dict(color=COLORS['slate'],  width=1.2),
                                     stackgroup='aux'))
            fig.update_layout(**{**PLOT_LAYOUT,
                'title': dict(text='Auxiliary Power Breakdown (kW)', font=dict(size=10)),
                'yaxis': dict(title='kW', gridcolor='#1e2433', tickfont=dict(size=9))})
            st.plotly_chart(fig, width="stretch")

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Identified Physical Parameters</div>',
                unsafe_allow_html=True)

    if st.session_state.models:
        sp = st.session_state.models['SOLVED_PARAMS']
        tp = st.session_state.models['tin_params']
        ap = st.session_state.models['aux_params']

        params_display = [
            ("i_n",          f"{sp[0]:.4e} A/m²",              "Internal current density loss"),
            ("i₀",           f"{sp[1]:.4e} A/m²",             "Exchange current density (80°C ref)"),
            ("α",            f"{sp[2]:.4f}",                  "Charge transfer coefficient"),
            ("k_λ",          f"{sp[3]:.4f}",                  "Membrane hydration scaling"),
            ("m_trans",      f"{sp[4]:.4e} V",          "Mass transport scaling"),
            ("n_growth",     f"{sp[5]:.4e} m²/A",                 "Mass transport amplification"),
            ("k_dry",        f"{sp[6]:.4e} 1/K",                 "Thermal drying coefficient"),
            ("hA",           f"{sp[7]:.1f} W/K",             "Convective heat conductance"),
            ("heat_gain",    f"{tp['heat_gain']:.4e} K/(W·s)",       "T_in heat gain coefficient"),
            ("cool_base",    f"{tp['cooling_base']:.4e} Hz",    "T_in baseline cooling"),
            ("cool_wp",      f"{tp['cooling_wp']:.4e} Hz/rps",      "Water pump cooling coeff"),
            ("cool_bp",      f"{tp['cooling_bp']:.4e} Hz/(kg/s)",   "Bypass flow cooling coeff"),
            ("cool_ci",      f"{tp['cooling_interact']:.4e} 1/(rps·kg)",      "Pump-bypass interaction"),
            ("P_ref",        f"{ap['P_ref']:.0f} W",         "Compressor reference power"),
            ("N_ref",        f"{ap['N_ref']:.0f} RPM",       "Compressor reference speed"),
            ("aux_frac",     f"{ap['AUX_SMALL_FRACTION']*100:.2f}%", "Small pump fraction"),
            ("N_cells",      "330",                           "Stack cell count"),
        ]

        PARAM_DEFINITIONS = {
            "i_n": {
                "name": "Internal Current Density Loss",
                "definition": "Current lost to hydrogen crossing the membrane and reacting "
                               "internally, without producing usable output. Keeps the "
                               "activation term from blowing up at zero external load.",
                "equation": "V_act = (RT / αnF) · ln[(i + i_n) / i₀]",
            },
            "i₀": {
                "name": "Exchange Current Density",
                "definition": "The reaction's baseline willingness to proceed at equilibrium "
                               "(80°C reference, Arrhenius-corrected for actual stack "
                               "temperature). Higher i₀ means a smaller activation voltage "
                               "drop at low current.",
                "equation": "V_act = (RT / αnF) · ln[(i + i_n) / i₀]",
            },
            "α": {
                "name": "Charge Transfer Coefficient",
                "definition": "How symmetric the reaction's energy barrier is between the "
                               "forward and reverse direction. Sets the slope of the "
                               "activation voltage drop.",
                "equation": "V_act = (RT / αnF) · ln[(i + i_n) / i₀]",
            },
            "k_λ": {
                "name": "Membrane Hydration Scaling",
                "definition": "How strongly current density re-hydrates the membrane through "
                               "water produced by the reaction. Drives the membrane's "
                               "steady-state water content, which sets conductivity.",
                "equation": "λ_ss = λ_base + k_λ·(i / 12000) − k_dry·(T − 353.15)",
            },
            "m_trans": {
                "name": "Mass Transport Scaling",
                "definition": "Sets the size of the voltage crash at high current, when "
                               "reactants can't diffuse to the reaction sites fast enough.",
                "equation": "V_conc = m_trans · (e^(n_growth·i) − 1)",
            },
            "n_growth": {
                "name": "Mass Transport Amplification",
                "definition": "Controls how suddenly the high-current voltage crash kicks in, "
                               "how close the stack is to its limiting current before losses "
                               "accelerate.",
                "equation": "V_conc = m_trans · (e^(n_growth·i) − 1)",
            },
            "k_dry": {
                "name": "Thermal Drying Coefficient",
                "definition": "The counterpart to k_λ, heat drives water out of the membrane "
                               "over time, hurting conductivity as stack temperature climbs "
                               "above the 80°C reference.",
                "equation": "λ_ss = λ_base + k_λ·(i / 12000) − k_dry·(T − 353.15)",
            },
            "hA": {
                "name": "Convective Heat Conductance",
                "definition": "How efficiently heat moves from the stack into the coolant "
                               "loop. Sets how fast stack temperature responds to the gap "
                               "between stack and coolant inlet temperature.",
                "equation": "Q_conv = hA · (T_stack − T_in)",
            },
            "heat_gain": {
                "name": "T_in Heat Gain Coefficient",
                "definition": "How much of the electrical power request shows up as heat "
                               "entering the coolant inlet thermal model per second.",
                "equation": "T[i] = T[i−1] + (heat_gain·P − k_cool·(T[i−1] − T_amb))·dt",
            },
            "cool_base": {
                "name": "T_in Baseline Cooling",
                "definition": "The fixed portion of coolant loop cooling that applies "
                               "regardless of pump speed or bypass flow.",
                "equation": "k_cool = cool_base + cool_wp·wp^0.8 + cool_bp·bp + cool_ci·wp·bp",
            },
            "cool_wp": {
                "name": "Water Pump Cooling Coefficient",
                "definition": "How much additional cooling the coolant inlet model gains "
                               "from increased coolant pump speed.",
                "equation": "k_cool = cool_base + cool_wp·wp^0.8 + cool_bp·bp + cool_ci·wp·bp",
            },
            "cool_bp": {
                "name": "Bypass Flow Cooling Coefficient",
                "definition": "How much additional cooling the coolant inlet model gains "
                               "from increased bypass valve flow, independent of pump speed.",
                "equation": "k_cool = cool_base + cool_wp·wp^0.8 + cool_bp·bp + cool_ci·wp·bp",
            },
            "cool_ci": {
                "name": "Pump-Bypass Interaction",
                "definition": "Captures the combined, non-additive cooling effect when both "
                               "pump speed and bypass flow increase together.",
                "equation": "k_cool = cool_base + cool_wp·wp^0.8 + cool_bp·bp + cool_ci·wp·bp",
            },
            "P_ref": {
                "name": "Compressor Reference Power",
                "definition": "The air compressor's power draw at its reference operating "
                               "speed, the anchor point for the cubic fan affinity power law.",
                "equation": "P_comp = P_ref · (N_comp / N_ref)³",
            },
            "N_ref": {
                "name": "Compressor Reference Speed",
                "definition": "The compressor RPM corresponding to P_ref, used to scale "
                               "compressor speed with air pressure ratio via the fan affinity "
                               "law.",
                "equation": "N_comp = N_ref · (P_air / P_atm)^0.5",
            },
            "aux_frac": {
                "name": "Small Pump Fraction",
                "definition": "The share of gross stack power consumed by small auxiliary "
                               "pumps (coolant, H2 recirculation), modeled as a fixed "
                               "fraction of gross output.",
                "equation": "P_aux_small = P_gross · aux_frac",
            },
            "N_cells": {
                "name": "Stack Cell Count",
                "definition": "The number of individual cells in series that make up the "
                               "TFCM2-F stack. Multiplies single-cell voltage to get total "
                               "stack voltage.",
                "equation": "V_stack = (V_n − V_act − V_ohmic − V_conc) · N_cells",
            },
        }

        param_cols = st.columns(4)
        for i, (sym, val, desc) in enumerate(params_display):
            with param_cols[i % 4]:
                st.markdown(f"""
                <div class="param-card">
                  <div class="param-label">{desc}</div>
                  <div style="display:flex;justify-content:space-between;
                              align-items:baseline;margin-top:4px;">
                    <span style="font-family:'IBM Plex Mono',monospace;
                                 font-size:11px;color:#475569;">{sym}</span>
                    <span class="param-value" style="font-size:13px;">{val}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

                defn = PARAM_DEFINITIONS.get(sym)
                if defn:
                    with st.popover("ⓘ definition", use_container_width=True):
                        st.markdown(f'<div class="param-def-heading">{defn["name"]}</div>',
                                    unsafe_allow_html=True)
                        st.markdown(f'<div class="param-def-body">{defn["definition"]}</div>',
                                    unsafe_allow_html=True)
                        st.markdown(f'<div class="param-def-eq">{defn["equation"]}</div>',
                                    unsafe_allow_html=True)


# ==============================================================================
# ROUTER
# ==============================================================================
if st.session_state.panel == 1:
    render_panel1()
elif st.session_state.panel == 2:
    render_panel2()
elif st.session_state.panel == 3:
    render_panel3()