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

/* ── Panel 3 KPI cards ──────────────────────────────────────────────────── */
.kpi-card {
    background: #141720;
    border: 1px solid #1e2433;
    border-top: 2px solid #f97316;
    border-radius: 6px;
    padding: 12px 16px;
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

/* ── Model architecture narrative ──────────────────────────────────────── */
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


# ── POLARIZATION CURVE BINNING HELPER ─────────────────────────────────────────
def compute_polarization_curve(i_arr, v_arr, num_bins):
    """
    Groups dynamic time-series current density and voltage into bins to extract 
    a clean steady-state polarization curve without dynamic zigzag artifacts.
    """
    df_pol = pd.DataFrame({'i': i_arr, 'v': v_arr})
    
    # Filter out near-zero or idle current points
    df_pol = df_pol[df_pol['i'] > 0.01].copy()
    
    if len(df_pol) == 0:
        return np.array([]), np.array([])
    
    # Bin the current density range
    bin_edges = np.linspace(df_pol['i'].min(), df_pol['i'].max(), num_bins + 1)
    df_pol['i_bin'] = pd.cut(df_pol['i'], bins=bin_edges, include_lowest=True)
    
    # Calculate the mean voltage per bin
    summary = df_pol.groupby('i_bin', observed=True).agg(
        i_mean=('i', 'mean'),
        v_mean=('v', 'mean')
    ).dropna().reset_index()
    
    return summary['i_mean'].values, summary['v_mean'].values


# ── FUEL CELL HTML ─────────────────────────────────────────────────────────────
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
      <stop offset="0%"   style="stop-color:#3a4358"/>
      <stop offset="100%" style="stop-color:#2b3346"/>
    </linearGradient>
    <linearGradient id="faceFront" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   style="stop-color:#242b3a"/>
      <stop offset="100%" style="stop-color:#181d29"/>
    </linearGradient>
    <linearGradient id="faceSide" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   style="stop-color:#181c27"/>
      <stop offset="100%" style="stop-color:#101319"/>
    </linearGradient>
    <pattern id="vent" width="7" height="7" patternTransform="rotate(45)" patternUnits="userSpaceOnUse">
      <rect width="7" height="7" fill="#12151d"/>
      <line x1="0" y1="0" x2="0" y2="7" stroke="#3a4358" stroke-width="2.2"/>
    </pattern>
    <radialGradient id="fanHub" cx="35%" cy="35%" r="70%">
      <stop offset="0%"   style="stop-color:#2d3748"/>
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

  <polygon points="380,110 648,265 514,343 246,188" fill="url(#faceTop)" stroke="#475569" stroke-width="1.5"/>
  <polygon points="380,210 648,365 648,265 380,110" fill="url(#faceFront)" stroke="#475569" stroke-width="1.5"/>
  <polygon points="648,365 514,443 514,343 648,265" fill="url(#faceSide)" stroke="#475569" stroke-width="1.5"/>

  <line x1="380" y1="210" x2="648" y2="265" stroke="#3a4358" stroke-width="2"/>
  <line x1="380" y1="110" x2="648" y2="365" stroke="#3a4358" stroke-width="2"/>

  <polygon points="387,153 494,215 454,238 347,176" fill="url(#vent)" stroke="#475569" stroke-width="1"/>

  <path d="M 610,290 C 690,278 750,310 828,292" fill="none" stroke="#f97316" stroke-width="11" stroke-linecap="round" opacity="0.92"/>
  <text x="836" y="288" font-family="IBM Plex Mono" font-size="10" fill="#f97316" font-weight="600">HV+</text>
  <text x="836" y="302" font-family="IBM Plex Mono" font-size="10" fill="#f97316" font-weight="600">HV−</text>

  <circle cx="300" cy="303" r="38" fill="url(#fanHub)" stroke="#3a4358" stroke-width="2.5"/>
  <g transform="translate(300,303)">
    <animateTransform attributeName="transform" type="rotate" values="0 0 0;360 0 0" dur="1.8s" repeatCount="indefinite" additive="sum"/>
    <path d="M0,-29 Q13,-13 0,0 Q-13,-13 0,-29"  fill="#38bdf8" opacity="0.75"/>
    <path d="M29,0  Q13,13  0,0 Q13,-13  29,0"   fill="#38bdf8" opacity="0.75"/>
    <path d="M0,29  Q-13,13 0,0 Q13,13   0,29"   fill="#38bdf8" opacity="0.75"/>
    <path d="M-29,0 Q-13,-13 0,0 Q-13,13 -29,0"  fill="#38bdf8" opacity="0.75"/>
  </g>

  <rect x="418" y="132" width="98" height="22" rx="3" fill="#0d0f14" opacity="0.55"/>
  <text x="467" y="147" text-anchor="middle" font-family="IBM Plex Mono" font-size="11" fill="#94a3b8" font-weight="600" letter-spacing="0.1em">330 CELLS</text>

  <line x1="55" y1="150" x2="332" y2="195" stroke="#4ade80" stroke-width="3" marker-end="url(#aH2)" stroke-dasharray="7 4">
    <animate attributeName="stroke-dashoffset" values="22;0" dur="0.8s" repeatCount="indefinite"/>
  </line>
  <text x="30" y="132" font-family="IBM Plex Mono" font-size="13" fill="#4ade80" font-weight="600">H₂</text>

  <line x1="55" y1="368" x2="278" y2="318" stroke="#38bdf8" stroke-width="2.6" marker-end="url(#aAir)" stroke-dasharray="7 4">
    <animate attributeName="stroke-dashoffset" values="22;0" dur="1.1s" repeatCount="indefinite"/>
  </line>
  <text x="14" y="396" font-family="IBM Plex Mono" font-size="13" fill="#38bdf8" font-weight="600">AIR</text>

  <text x="460" y="55" text-anchor="middle" font-family="IBM Plex Mono" font-size="18" fill="#f1f5f9" font-weight="600" letter-spacing="0.16em">TFCM2-F</text>
  <text x="460" y="547" text-anchor="middle" font-family="IBM Plex Mono" font-size="9" fill="#475569" letter-spacing="0.06em">
    1270 × 630 × 410 mm  |  Toyota Fuel Cell Stack Module
  </text>
</svg>
</body></html>"""


# ── PLOT HELPER ───────────────────────────────────────────────────────────────
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


# ── SESSION STATE INITIALIZATION ──────────────────────────────────────────────
_defaults = {
    'panel'         : 1,
    'df_input'      : None,
    'dt'            : 0.1,
    't_start_K'     : 295.15,
    'results'       : None,
    'has_real_data' : False,
    'input_mode'    : 'csv',
    'models'        : None,
    'model_error'   : None,
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


# ── BLOCK EDITOR (Drive Cycle Builder) ─────────────────────────────────────────
def render_block_editor():
    st.markdown('<div class="panel-title">VehicleSpy Script Builder</div>', unsafe_allow_html=True)
    st.caption("Build custom drive cycle power profiles step-by-step.")
    
    if 'blocks' not in st.session_state:
        st.session_state.blocks = [
            {'type': 'Ramp', 'p_start': 0.0, 'p_end': 20.0, 'duration': 10.0},
            {'type': 'Hold', 'p_start': 20.0, 'p_end': 20.0, 'duration': 15.0},
            {'type': 'Ramp', 'p_start': 20.0, 'p_end': 50.0, 'duration': 10.0},
            {'type': 'Hold', 'p_start': 50.0, 'p_end': 50.0, 'duration': 20.0},
            {'type': 'Ramp', 'p_start': 50.0, 'p_end': 0.0, 'duration': 10.0},
        ]
    
    updated_blocks = []
    for idx, blk in enumerate(st.session_state.blocks):
        cols = st.columns([0.8, 2, 2, 2, 1.5])
        with cols[0]:
            st.markdown(f"**#{idx+1}**")
        with cols[1]:
            b_type = st.selectbox(f"Type ##{idx}", ["Hold", "Ramp"], index=0 if blk['type'] == "Hold" else 1, key=f"btype_{idx}", label_visibility="collapsed")
        with cols[2]:
            p_start = st.number_input(f"Start (kW) ##{idx}", value=float(blk['p_start']), step=5.0, key=f"pstart_{idx}")
        with cols[3]:
            p_end = p_start if b_type == "Hold" else st.number_input(f"End (kW) ##{idx}", value=float(blk['p_end']), step=5.0, key=f"pend_{idx}")
        with cols[4]:
            dur = st.number_input(f"Sec ##{idx}", value=float(blk['duration']), min_value=1.0, step=5.0, key=f"dur_{idx}")
        
        updated_blocks.append({'type': b_type, 'p_start': p_start, 'p_end': p_end, 'duration': dur})
    
    st.session_state.blocks = updated_blocks
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("➕ Add Step", use_container_width=True):
            st.session_state.blocks.append({'type': 'Hold', 'p_start': 10.0, 'p_end': 10.0, 'duration': 10.0})
            st.rerun()
    with col_b:
        if len(st.session_state.blocks) > 1 and st.button("➖ Remove Last", use_container_width=True):
            st.session_state.blocks.pop()
            st.rerun()
            
    dt = st.session_state.dt
    time_series = []
    p_series = []
    curr_t = 0.0
    
    for blk in st.session_state.blocks:
        steps = int(np.round(blk['duration'] / dt))
        if steps < 1: steps = 1
        t_vals = np.linspace(curr_t, curr_t + blk['duration'], steps, endpoint=False)
        p_vals = np.linspace(blk['p_start'] * 1000.0, blk['p_end'] * 1000.0, steps)
        
        time_series.extend(t_vals)
        p_series.extend(p_vals)
        curr_t += blk['duration']
        
    df_built = pd.DataFrame({'time': time_series, 'power_request': p_series})
    st.session_state.df_input = df_built
    
    fig = go.Figure(go.Scatter(x=df_built['time'], y=df_built['power_request']/1000, mode='lines', line=dict(color=COLORS['orange'], width=1.5)))
    fig.update_layout(**{**PLOT_LAYOUT, 'height': 160, 'title': dict(text='Synthesized Power Profile (kW)', font=dict(size=10))})
    st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# PANEL 1 — INPUT CONFIGURATION
# ==============================================================================
@st.dialog("Model Architecture", width="large")
def _show_architecture_dialog():
    _narrative = [
        ("01 · THE REQUEST ARRIVES", "The Request Arrives",
         "Every simulation starts as one number changing over time: power request. "
         "That's all a drive cycle really is—a target the stack has to chase."),
        ("02 · GUESSING THE SENSORS", "Guessing the Sensors",
         "The first thing the twin does is stand in for six sensors it doesn't have. "
         "Random Forest and linear models predict current density, air pressure, "
         "bypass flow, and coolant pump speed."),
        ("03 · WAITING FOR THE COOLANT", "Waiting for the Coolant to Catch Up",
         "A nonlinear state-space integrator tracks coolant inlet temperature second by second, "
         "balancing heat gained from load against cooling supplied."),
        ("04 · THE PHYSICS CORE", "The Physics Core Does the Real Work",
         "Nernst voltage sets the ceiling, activation and ohmic losses pull it down based on "
         "membrane hydration (Springer's model) and temperature."),
        ("05 · WHAT PHYSICS MISSES", "The Physics Model Admits What It Doesn't Know",
         "An SVR trained on residual gaps corrects physics predictions to match physical hardware."),
        ("06 · POWER THE STACK KEEPS", "Turning Voltage Into Power the Stack Actually Keeps",
         "Parasitic loads (compressor, coolant pump, H₂ pump, DC-DC) are subtracted to compute net power."),
        ("07 · COUNTING THE HYDROGEN", "Counting the Hydrogen",
         "Faraday's law plus ML residual tracking computes exact hydrogen consumption.")
    ]
    for step_label, heading, body in _narrative:
        st.markdown(f"""
        <div class="narrative-block">
          <div class="narrative-step">{step_label}</div>
          <div class="narrative-heading">{heading}</div>
          <div class="narrative-body">{body}</div>
        </div>""", unsafe_allow_html=True)


def render_panel1():
    st.markdown('<div class="panel-title">Toyota KAUST Clean Combustion Research Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">TFCM2-F Digital Twin</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Physics-ML Hybrid Stack Simulation</div>', unsafe_allow_html=True)

    if st.session_state.model_error:
        st.error(f"**Model files not found.** Run your notebook first to generate all .pkl files.\n\n`{st.session_state.model_error}`")
        return

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)

    left, mid, right = st.columns([1.1, 1.5, 1.0], gap="large")

    with left:
        st.markdown('<div class="panel-title">Ambient Conditions</div>', unsafe_allow_html=True)
        t_room = st.slider("Starting Room Temperature (°C)", 15.0, 45.0, 22.0, 0.5)
        rh_val = st.slider("Relative Humidity", 0.0, 1.0, 0.48, 0.01, format="%.2f")
        st.session_state.t_start_K = t_room + 273.15

    with mid:
        st.markdown('<div class="panel-title">Drive Cycle Input</div>', unsafe_allow_html=True)
        mode = st.radio("Drive cycle input mode", ["Upload CSV", "Drive Cycle Builder"], horizontal=True, label_visibility="collapsed")

        if mode == "Upload CSV":
            st.session_state.input_mode = 'csv'
            f = st.file_uploader("VehicleSpy CSV export", type=["csv"], label_visibility="collapsed")
            if f:
                try:
                    from model_core import process_fc_data_from_upload
                    df, dt, t_s = process_fc_data_from_upload(f)
                    st.session_state.df_input      = df
                    st.session_state.dt            = dt
                    st.session_state.has_real_data = True
                    st.success(f"Loaded {len(df):,} rows  |  dt={dt:.3f}s  |  T_start={t_s-273.15:.1f}°C")
                    
                    fig = go.Figure(go.Scatter(x=df['time'], y=df['power_request']/1000, mode='lines', line=dict(color=COLORS['orange'], width=1.2)))
                    fig.update_layout(**{**PLOT_LAYOUT, 'height': 160, 'title': dict(text='Power Request (kW)', font=dict(size=10))})
                    st.plotly_chart(fig, use_container_width=True)
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
        st.markdown('<p style="font-size:12px;color:#64748b;line-height:1.5;">Physics-ML hybrid twin combining a convective thermal engine, six ECU sub-models, and an SVR residual correction layer.</p>', unsafe_allow_html=True)
        if st.button("📖 Learn more about the model architecture", use_container_width=True):
            _show_architecture_dialog()

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)
    components.html(_fc_html(width=820, height=440), height=444, scrolling=False)

    ready = (st.session_state.df_input is not None and st.session_state.models is not None)

    btn_l, btn_c, btn_r = st.columns([1.4, 1, 1.4])
    with btn_c:
        if ready:
            n   = len(st.session_state.df_input)
            dur = n * st.session_state.dt
            mode_badge = 'READY · CSV MODE' if st.session_state.has_real_data else 'READY · PREDICT MODE'
            badge_class = 'badge-active' if st.session_state.has_real_data else 'badge-predict'
            st.markdown(f'<div style="text-align:center"><div class="{badge_class}">{mode_badge}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<p style="text-align:center;font-size:11px;color:#64748b;font-family:\'IBM Plex Mono\',monospace;margin-top:2px;">{n:,} steps · {dur:.1f}s cycle</p>', unsafe_allow_html=True)
            
            if st.button("🚀 INITIALIZE SIMULATION RUN", use_container_width=True):
                st.session_state.panel = 2
                st.rerun()
        else:
            st.button("🚀 INITIALIZE SIMULATION RUN", disabled=True, use_container_width=True)


# ==============================================================================
# PANEL 2 — SIMULATION PROGRESS
# ==============================================================================
def render_panel2():
    st.markdown('<div class="panel-title">Toyota KAUST Clean Combustion Research Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Simulating Digital Twin...</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Executing electrothermal physics core & SVR correction models</div>', unsafe_allow_html=True)
    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)

    components.html(_fc_html(width=820, height=440, grow=True), height=444, scrolling=False)

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.markdown('<p style="text-align:center;font-family:\'IBM Plex Mono\',monospace;color:#f97316;">[1/3] Initializing state-space integrators...</p>', unsafe_allow_html=True)
    progress_bar.progress(25)
    time.sleep(0.2)

    status_text.markdown('<p style="text-align:center;font-family:\'IBM Plex Mono\',monospace;color:#f97316;">[2/3] Executing physics-ML coupled solver...</p>', unsafe_allow_html=True)
    progress_bar.progress(60)

    try:
        from model_core import run_digital_twin_simulation
        results = run_digital_twin_simulation(
            st.session_state.df_input,
            st.session_state.models,
            dt=st.session_state.dt,
            t_start_K=st.session_state.t_start_K
        )
        st.session_state.results = results
        progress_bar.progress(100)
        status_text.markdown('<p style="text-align:center;font-family:\'IBM Plex Mono\',monospace;color:#4ade80;">[3/3] Simulation Complete!</p>', unsafe_allow_html=True)
        time.sleep(0.3)
        st.session_state.panel = 3
        st.rerun()
    except Exception as e:
        st.error(f"Simulation Execution Failed: {e}")
        if st.button("⬅ Return to Input Panel"):
            st.session_state.panel = 1
            st.rerun()


# ==============================================================================
# PANEL 3 — RESULTS DASHBOARD
# ==============================================================================
def render_panel3():
    results = st.session_state.results
    if results is None:
        st.session_state.panel = 1
        st.rerun()
        return

    # Header Row
    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.markdown('<div class="panel-title">Toyota KAUST Clean Combustion Research Center</div>', unsafe_allow_html=True)
        st.markdown('<div class="main-title">Digital Twin Simulation Results</div>', unsafe_allow_html=True)
    with top_r:
        if st.button("🔄 Reset / New Simulation", use_container_width=True):
            st.session_state.panel = 1
            st.session_state.results = None
            st.rerun()

    st.markdown('<hr class="h-rule">', unsafe_allow_html=True)

    # Executive KPI Cards
    p_net_kw = results['power_net'] / 1000.0
    p_peak = np.max(p_net_kw)
    h2_tot = results['h2_cum_g'][-1] if 'h2_cum_g' in results else 0.0
    v_pred = results['V_final']

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Peak Net Power</div><div class="kpi-value">{p_peak:.1f} kW</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">H₂ Consumed</div><div class="kpi-value">{h2_tot:.1f} g</div></div>', unsafe_allow_html=True)
    with k3:
        avg_v = np.mean(v_pred)
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Mean Voltage</div><div class="kpi-value">{avg_v:.1f} V</div></div>', unsafe_allow_html=True)
    with k4:
        if st.session_state.has_real_data and 'V_real' in results:
            rmse = np.sqrt(np.mean((results['V_real'] - v_pred)**2)) * 1000.0 # mV
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Voltage RMSE</div><div class="kpi-value">{rmse:.1f} mV</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">Simulation Mode</div><div class="kpi-value">Predictive</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    # Dashboard Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 System Overview", 
        "📈 Polarization Curve", 
        "🔧 Subsystem Diagnostics", 
        "📥 Export Results"
    ])

    t = results['time']

    # ── TAB 1: OVERVIEW ────────────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            t_power = [(results['power_request']/1000.0, 'Request', COLORS['orange'], 'solid'),
                       (results['power_net']/1000.0, 'Net Power', COLORS['green'], 'solid')]
            st.plotly_chart(mini_plot(t, t_power, 'Power Trajectory (kW)', 'kW'), use_container_width=True)

            t_temp = [(results['T_stack'] - 273.15, 'Stack Temp', COLORS['purple'], 'solid')]
            st.plotly_chart(mini_plot(t, t_temp, 'Stack Temperature (°C)', '°C'), use_container_width=True)

        with col_b:
            t_volt = [(v_pred, 'Digital Twin', PREDICTED_COLOR, 'solid')]
            if st.session_state.has_real_data and 'V_real' in results:
                t_volt.append((results['V_real'], 'Measured', MEASURED_COLOR, 'dash'))
            st.plotly_chart(mini_plot(t, t_volt, 'Stack Voltage (V)', 'V'), use_container_width=True)

            t_curr = [(results['current_density'], 'Current Density', COLORS['blue'], 'solid')]
            st.plotly_chart(mini_plot(t, t_curr, 'Current Density', 'A/cm²'), use_container_width=True)

    # ── TAB 2: POLARIZATION CURVE (BINNED & SLIDER INTEGRATED) ─────────────────
    with tab2:
        st.subheader("Stack Polarization Curve")
        st.caption("Extracted by binning dynamic time-series current density and averaging stack voltage to eliminate transient hysteresis and sawtooth artifacts.")

        pol_bins = st.slider(
            "Curve Smoothness (Number of Bins)", 
            min_value=10, 
            max_value=150, 
            value=40, 
            step=10, 
            help="Lower values produce a smoother curve; higher values show more local detail."
        )

        i_pred = results['current_density']
        v_twin = results['V_final']

        i_twin_bin, v_twin_bin = compute_polarization_curve(i_pred, v_twin, num_bins=pol_bins)

        fig_pol = go.Figure()

        # Digital Twin Binned Trace
        fig_pol.add_trace(go.Scatter(
            x=i_twin_bin,
            y=v_twin_bin,
            mode='lines+markers',
            name='Digital Twin',
            line=dict(color=PREDICTED_COLOR, width=3),
            marker=dict(size=5)
        ))

        # Measured Binned Trace (if real data available)
        if st.session_state.has_real_data and 'V_real' in results:
            i_real = results.get('current_density_real', i_pred)
            v_real = results['V_real']
            
            i_real_bin, v_real_bin = compute_polarization_curve(i_real, v_real, num_bins=pol_bins)
            
            fig_pol.add_trace(go.Scatter(
                x=i_real_bin,
                y=v_real_bin,
                mode='lines+markers',
                name='Measured (VehicleSpy)',
                line=dict(color=MEASURED_COLOR, width=3, dash='dash'),
                marker=dict(size=5)
            ))

        fig_pol.update_layout(
            xaxis_title='Current Density',
            yaxis_title='Stack Voltage (V)',
            template='plotly_dark',
            paper_bgcolor='#141720',
            plot_bgcolor='#0d0f14',
            font=dict(family='IBM Plex Mono', size=10, color='#94a3b8'),
            hovermode='x unified',
            height=380,
            margin=dict(l=40, r=40, t=40, b=40)
        )

        st.plotly_chart(fig_pol, use_container_width=True)

    # ── TAB 3: DIAGNOSTICS ─────────────────────────────────────────────────────
    with tab3:
        d1, d2 = st.columns(2)
        with d1:
            if 'p_air' in results:
                t_air = [(results['p_air'], 'Air Pressure', COLORS['teal'], 'solid')]
                st.plotly_chart(mini_plot(t, t_air, 'Air Manifold Pressure (kPa)', 'kPa'), use_container_width=True)
            if 'h2_rate' in results:
                t_h2 = [(results['h2_rate'], 'H2 Rate', COLORS['green'], 'solid')]
                st.plotly_chart(mini_plot(t, t_h2, 'H₂ Consumption Rate (g/s)', 'g/s'), use_container_width=True)
        with d2:
            if 'svr_residual' in results:
                t_svr = [(results['svr_residual'], 'SVR Delta', COLORS['yellow'], 'solid')]
                st.plotly_chart(mini_plot(t, t_svr, 'SVR Residual Voltage Correction (V)', 'V'), use_container_width=True)

    # ── TAB 4: EXPORT ──────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Simulation Results Data Table")
        df_export = pd.DataFrame({
            'Time (s)': t,
            'Power Request (W)': results['power_request'],
            'Net Power (W)': results['power_net'],
            'Stack Voltage (V)': v_pred,
            'Current Density': results['current_density'],
            'Stack Temp (K)': results['T_stack']
        })
        st.dataframe(df_export.head(100), use_container_width=True)
        csv_bytes = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Simulation CSV",
            data=csv_bytes,
            file_name="tfcm2f_digital_twin_results.csv",
            mime="text/csv"
        )


# ==============================================================================
# MAIN ROUTER
# ==============================================================================
if st.session_state.panel == 1:
    render_panel1()
elif st.session_state.panel == 2:
    render_panel2()
elif st.session_state.panel == 3:
    render_panel3()