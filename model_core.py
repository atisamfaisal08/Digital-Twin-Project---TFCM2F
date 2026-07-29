# ==============================================================================
# model_core.py — TFCM2-F Digital Twin: All Physics & Inference Functions
# Drop this file in the same directory as app.py
# Run your notebook first to generate the .pkl files, then launch the GUI
# ==============================================================================

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# ── PHYSICAL CONSTANTS (Cell 1) ───────────────────────────────────────────────
F          = 96485.0
R          = 8.314
n_electrons= 2.0
E0         = 1.229
N_cells    = 330.0
TAU_THERMAL= 30.0
TAU_LAMBDA = 5.0
LAB_RH     = 0.48
tm         = 8.5e-6
x_O2       = 0.21
M_stack    = 56.0
Cp_stack   = 500.0
E_th       = 1.48
Area       = 0.0297
M_H2       = 2.016e-3   # kg/mol
F_const    = 96485.0
P_atm      = 101325.0

# ── COOLANT PROPERTIES (grounded in real FC stack coolant spec, replacing
# the old fixed cp_coolant=3450.0 and rho_coolant=1000.0 approximations) ────
COOLANT_DENSITY = 1065.0  # kg/m^3, was 1000.0 (plain water); now matches the
                           # actual coolant, same 1.065 figure already used
                           # in CSV ingestion

_CP_TABLE_TEMP_C = np.array([-30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
_CP_TABLE_KJ     = np.array([3.23, 3.25, 3.28, 3.3, 3.32, 3.38, 3.42,
                              3.46, 3.49, 3.51, 3.55, 3.58, 3.6])

def get_cp_coolant(T_kelvin):
    """Interpolated coolant specific heat, J/(kg*K). Was a fixed 3450.0
    constant; real value varies ~3230-3600 J/(kg*K) across the stack's
    actual 20-90C operating range."""
    T_C = T_kelvin - 273.15
    return np.interp(T_C, _CP_TABLE_TEMP_C, _CP_TABLE_KJ) * 1000.0

# ── EMPIRICAL PUMP FLOW MODEL (replaces the old fixed pump_disp assumption).
# The physical coolant pump is centrifugal, not positive-displacement, so
# flow rate is NOT a fixed displacement x RPM. These coefficients were fit
# by back-calculating real effective mass flow from energy-balance data
# (Q_conv / (cp * delta_T)), filtered to delta_T >= 5K, then regressed
# against real wp_rps and bypass_flow. Validated at R^2=0.9433. ────────────
PUMP_COEF_WP   = 0.035344
PUMP_COEF_BP   = 0.3025
PUMP_INTERCEPT = -0.1364

def estimate_mdot(wp_rps, bypass_flow):
    """Empirical mass flow, kg/s. Clipped at a small positive floor since
    the fitted intercept is slightly negative and this was only validated
    within the wp_rps=10-35 range it was trained on."""
    m = PUMP_COEF_WP * wp_rps + PUMP_COEF_BP * bypass_flow + PUMP_INTERCEPT
    return max(m, 1e-3)

# ── ELECTROCHEMISTRY (Cell 1) ─────────────────────────────────────────────────
def calculate_lambda_base(rh):
    return 0.043 + 17.81*rh - 39.85*(rh**2) + 36.0*(rh**3)

def nernst_voltage(T, P_H2_pa, P_air_pa, P_H2O_pa):
    P_H2_atm  = P_H2_pa  / P_atm
    P_air_atm = P_air_pa / P_atm
    P_H2O_atm = P_H2O_pa / P_atm
    P_O2_atm  = x_O2 * (P_air_atm - P_H2O_atm)
    inner_log = (P_H2_atm * P_O2_atm**0.5) / np.maximum(P_H2O_atm, 1e-6)
    return E0 + (R * T / (n_electrons * F)) * np.log(np.maximum(inner_log, 1e-10))

def cell_voltage(i, T, P_H2, P_air, P_H2O, params, l_dynamic):
    i_n, i_0, alpha, k_lambda, m_trans, n_growth, k_dry = params
    sigma   = (0.005139 * l_dynamic - 0.00326) * np.exp(1268 * (1/303 - 1/T)) * 100.0
    V_ohmic = i * (tm / np.maximum(sigma, 1e-6))
    V_n     = nernst_voltage(T, P_H2, P_air, P_H2O)
    V_act   = (R * T / (alpha * n_electrons * F)) * np.log((i + i_n) / np.maximum(i_0, 1e-12))
    V_conc  = m_trans * (np.exp(n_growth * i) - 1)
    return (V_n - V_act - V_ohmic - V_conc) * N_cells

# ── PHYSICS LOOP (Cell 3) ─────────────────────────────────────────────────────
def predict_voltage_loop(params, df_target, dt_target, t_start_target, return_coolant_split=False):
    hA = params[7]
    l_base = calculate_lambda_base(LAB_RH)
    T_stack_curr = t_start_target

    i_arr   = df_target['current_density'].values
    bp_arr  = df_target['bypass_flow'].values
    wp_arr  = df_target['wp_rps'].values
    h2_arr  = df_target['h2_p'].values
    air_arr = df_target['air_p'].values
    T_inlet = df_target['T_in_K'].values

    V_pred      = np.zeros(len(i_arr))
    T_stack_out = np.zeros(len(i_arr))
    T_out_out   = np.zeros(len(i_arr))

    T_ref = 353.15
    E_act = 40000.0
    R_gas = 8.314

    for k in range(len(i_arr)):
        current_params = params[0:7].copy()
        temp_factor = np.exp((-E_act / R_gas) * ((1.0 / T_stack_curr) - (1.0 / T_ref)))
        current_params[1] = params[1] * temp_factor

        k_lambda = current_params[3]
        k_dry    = current_params[6]
        l_steady_state = l_base + (k_lambda * (i_arr[k] / 12000.0)) - (k_dry * (T_stack_curr - 353.15))
        l_steady_state = np.clip(l_steady_state, 0.5, 14.0)

        T_C   = T_stack_curr - 273.15
        P_h2o = (10 ** (8.07131 - (1730.63 / (233.426 + T_C)))) * 133.322
        P_h2o = np.minimum(P_h2o, air_arr[k] * 0.9)

        V_val = cell_voltage(i_arr[k], T_stack_curr, h2_arr[k], air_arr[k], P_h2o, current_params, l_steady_state)

        cp_coolant_now = get_cp_coolant(T_inlet[k])
        m_dot     = estimate_mdot(wp_arr[k], bp_arr[k])
        C_coolant = np.maximum(m_dot * cp_coolant_now, 1e-3)
        Q_conv    = hA * (T_stack_curr - T_inlet[k])
        Q_gen     = (i_arr[k] * Area) * (E_th - (V_val / N_cells)) * N_cells
        dT_stack  = (Q_gen - Q_conv) * dt_target / (M_stack * Cp_stack)
        T_stack_curr += dT_stack
        T_fluid_out   = T_inlet[k] + (Q_conv / C_coolant)

        V_pred[k]      = V_val
        T_stack_out[k] = T_stack_curr
        T_out_out[k]   = T_fluid_out

    if return_coolant_split:
        return V_pred, T_stack_out, T_out_out
    else:
        return V_pred, T_out_out

# ── T_in THERMAL MODEL (Cell 10) ─────────────────────────────────────────────
def predict_tin_actuator_dynamic_nonlinear(df, t_start, heat_gain, cooling_base,
                                            cooling_wp, cooling_bp, cooling_interact,
                                            fixed_dt=None, T_amb=295.15):
    power = df['power_request'].values
    wp    = df['wp_rps'].values
    bp    = df['bypass_flow'].values
    n     = len(power)
    T     = np.zeros(n)
    T[0]  = t_start

    dts = np.full(n - 1, fixed_dt) if fixed_dt is not None else np.diff(df['time'].values)

    for i in range(1, n):
        dt     = dts[i - 1]
        T_prev = T[i - 1]
        Q_heat = heat_gain * power[i - 1]
        k_cool = (cooling_base
                  + cooling_wp    * (wp[i - 1] ** 0.8)
                  + cooling_bp    *  bp[i - 1]
                  + cooling_interact * wp[i - 1] * bp[i - 1])
        if k_cool < 1e-6:
            k_cool = 1e-6
        T[i] = T_prev + (Q_heat - k_cool * (T_prev - T_amb)) * dt
    return T

# ── AUXILIARY POWER (Cell 12) ─────────────────────────────────────────────────
def compute_net_power(p_gross, air_pressure_pa, N_ref, P_ref, AUX_SMALL_FRACTION=0.0136):
    comp_rpm    = N_ref * ((air_pressure_pa / P_atm) ** 0.5)
    p_comp      = P_ref * (comp_rpm / N_ref) ** 3
    p_aux_small = p_gross * AUX_SMALL_FRACTION
    p_converter = p_gross * 0.02
    p_net       = p_gross - p_comp - p_aux_small - p_converter
    return p_net, p_comp, p_aux_small, p_converter

# ── H2 CONSUMPTION (Cell 13) ─────────────────────────────────────────────────
def faraday_h2(current_density_arr):
    return (current_density_arr * Area * N_cells * M_H2 / (2 * F_const)) * 1e6

def build_h2_features(df, faraday_arr):
    slope_i = np.diff(df['current_density'].values, prepend=df['current_density'].values[0])
    slope_p = np.diff(df['power_request'].values,   prepend=df['power_request'].values[0])
    return np.column_stack([
        faraday_arr,
        df['current_density'].values,
        df['power_request'].values,
        slope_i,
        slope_p,
        df['air_p'].values,
        df['T_in_K'].values,
    ])

# ── ECU FEATURE BUILDERS (Cells 5-9) ─────────────────────────────────────────
def build_ecu_features(df):
    power  = df['power_request'].values
    slope  = np.diff(power, prepend=power[0])
    t_init = np.full(len(df), df['T_in_K'].values[0])
    return np.column_stack([power, slope, t_init])

def build_ecu_features_air(df):
    power  = df['power_request'].values
    slope  = np.diff(power, prepend=power[0])
    t_init = np.full(len(df), df['T_in_K'].values[0])
    return np.column_stack([power, slope, t_init])

def build_ecu_features_bp(df):
    power  = df['power_request'].values
    slope  = np.diff(power, prepend=power[0])
    t_init = np.full(len(df), df['T_in_K'].values[0])
    return np.column_stack([power, slope, t_init])

def build_ecu_features_wp(df):
    power  = df['power_request'].values
    slope  = np.diff(power, prepend=power[0])
    t_init = np.full(len(df), df['T_in_K'].values[0])
    return np.column_stack([power, slope, t_init])

def smooth_signal(arr, window=30):
    return pd.Series(arr).rolling(window=window, center=True, min_periods=1).mean().values

# ── DATA REFINERY (Cell 2) ────────────────────────────────────────────────────
# ── CSV COLUMN NAME SCHEMAS ───────────────────────────────────────────────────
# Two different tools export drive cycle logs with different header names for
# the exact same physical quantities. Each entry below maps one INTERNAL field
# name to the raw CSV column name for that schema. Add a third schema later by
# adding a third dict here, nothing else needs to change, process_fc_data_from
# _upload() auto-detects which schema is present and reads through this map.
COLUMN_SCHEMA_VSPY = {
    'name'            : 'VehicleSpy export',
    'time'            : 'Time',
    'voltage'         : 'Output Voltage',
    'current'         : 'Output Current',
    'bypass_flow'     : 'Bypass Flow Rate',      # raw unit: L/min
    'coolant_pump'    : 'Main Coolant Flow Rate',# raw unit: rpm
    'h2_pressure'     : 'H2 Intake Pressure',    # raw unit: kPa
    'air_pressure'    : 'Air Intake Pressure',   # raw unit: kPa
    'power_request'   : 'Power Request',         # raw unit: W
    'power_net'       : 'Net Power Output',      # raw unit: W
    'power_gross'     : 'Gross Power',           # raw unit: W (direct column)
    'coolant_in_temp' : 'Coolant Inlet Temp',    # raw unit: °C
    'coolant_out_temp': 'Coolant Outlet Temp',   # raw unit: °C
    'air_compressor'  : 'Air Compressor RPM',
    'h2_consumption'  : 'H2 Consumption',
}

COLUMN_SCHEMA_FC_A = {
    'name'            : 'FC_A logger export',
    'time'            : 'Time (abs)',
    # The actual exported CSV headers include the full "(Value [unit])"
    # suffix as literal text, not just the short code. Matching only the
    # short code ('VFC_A' instead of 'VFC_A (Value [V])') meant these columns
    # never matched at all, which was the real cause behind schema detection
    # struggling on files with mixed-looking naming.
    'voltage'         : 'VFC_A (Value [V])',
    'current'         : 'IFC_A (Value [A])',
    'bypass_flow'     : 'BYPAS_FLOW_A (Value [L/min])',   # same raw unit as VSpy
    'coolant_pump'    : 'WP_REV_A (Value [rpm])',         # same raw unit as VSpy
    'h2_pressure'     : 'SUP_PRE_FC_A (Value [kPa])',     # same raw unit as VSpy
    'air_pressure'    : 'PAFIC_A (Value [kPa])',          # same raw unit as VSpy
    'power_request'   : 'POWERREQ_A (Value [W])',
    'power_net'       : 'MES_FC_A (Value [W])',
    'power_gross'     : None,                             # no direct column, computed below
    'coolant_in_temp' : 'FCI_TEMP_A (Value [°C])',
    'coolant_out_temp': 'FCO_TEMP_A (Value [°C])',
    'air_compressor'  : 'ACPREV_A (Value [rpm])',
    'h2_consumption'  : 'CNSMH2_A (Value [mg])',
    # CONFIRMED (field-tested against a known cycle): CNSMH2_A is a rate
    # (mg/s per timestep), matching the same semantics as VSpy's H2
    # Consumption column, not a cumulative total. The earlier caution flagged
    # here has been resolved, no schema-specific handling is needed.
}

_KNOWN_SCHEMAS = [COLUMN_SCHEMA_VSPY, COLUMN_SCHEMA_FC_A]

# Every internal field this pipeline actually needs, used to drive per-field
# column resolution below. power_gross is deliberately absent from this list,
# it's handled separately since one schema computes it instead of reading it.
_REQUIRED_FIELDS = [
    'time', 'voltage', 'current', 'bypass_flow', 'coolant_pump',
    'h2_pressure', 'air_pressure', 'power_request', 'power_net',
    'coolant_in_temp', 'coolant_out_temp', 'air_compressor', 'h2_consumption',
]

# For most fields, checking schemas in _KNOWN_SCHEMAS order (VSpy first, then
# FC_A) doesn't matter, a file only ever has one of the two column names
# present for a given field. 'time' is the exception: some VehicleSpy
# exports include BOTH 'Time' (literal wall-clock time of day, e.g. 14:10)
# and 'Time (abs)' (a running stopwatch from cycle start) in the same file.
# The physics loop needs the stopwatch value for its dt calculation, wall
# clock time would silently break it. This override forces 'Time (abs)' to
# be checked first regardless of the default schema order, only falling
# back to 'Time' if 'Time (abs)' genuinely isn't present.
_FIELD_PREFERENCE_OVERRIDES = {
    'time': [COLUMN_SCHEMA_FC_A['time'], COLUMN_SCHEMA_VSPY['time']],
}


def _column_prefix(candidate):
    """Returns the stable part of a column name before its '(Value ...)'
    suffix, if it has one. Used as a fallback match target, since the unit
    text inside those brackets has been seen corrupted in real exports (a
    mangled degree symbol in *_TEMP_A columns specifically)."""
    idx = candidate.find(' (Value')
    return candidate[:idx] if idx != -1 else candidate


def _resolve_column(field_key, columns):
    """
    Finds the raw column name for one internal field by checking every known
    schema independently, rather than requiring one schema to match the whole
    file. This is what actually lets a genuinely mixed-naming CSV (some
    columns named the VSpy way, others the FC_A way — which is exactly what
    a real logging setup could produce during a partial migration between
    tools) still resolve correctly, instead of forcing an all-or-nothing
    choice between two whole schemas.

    Tries an exact match first. If nothing matches exactly, falls back to
    matching on the stable prefix before '(Value ...)', which handles real
    files where the unit text itself is corrupted or inconsistent (seen in
    practice: FCI_TEMP_A / FCO_TEMP_A headers with a mangled degree symbol
    instead of °C) without needing the exact byte sequence to be known ahead
    of time. Prefix matching is only attempted as a fallback, and only
    against the stable code portion, so it doesn't risk matching an
    unrelated column, e.g. this won't confuse the two BYPAS_FLOW_A columns
    some exports contain, since that field already matches exactly.
    """
    candidates = _FIELD_PREFERENCE_OVERRIDES.get(field_key)
    if candidates is None:
        candidates = [schema.get(field_key) for schema in _KNOWN_SCHEMAS
                      if schema.get(field_key) is not None]
    for col in candidates:
        if col in columns:
            return col
    for candidate in candidates:
        prefix = _column_prefix(candidate)
        if prefix == candidate:
            continue  # no '(Value ...)' suffix to be flexible about
        for col in columns:
            if col.startswith(prefix + ' (Value'):
                return col
    raise ValueError(
        f"Could not find a column for '{field_key}' in this CSV. "
        f"Expected one of: {', '.join(candidates)}.")


def process_fc_data_from_upload(uploaded_file):
    """
    Processes a drive cycle CSV upload into the model's internal format.
    Each required column is resolved independently against every known
    naming schema (VehicleSpy export, FC_A logger export), so files that mix
    naming conventions across columns still process correctly, and both
    naming conventions produce the exact same internal dataframe either way.
    """
    # Some exports contain invalid UTF-8 bytes (seen in practice: a mangled
    # degree symbol in temperature column headers like "FCI_TEMP_A (Value
    # [°C])"). Depending on the installed pandas version this can surface as
    # a confusing downstream error instead of a clear encoding failure, so
    # it's handled explicitly here: try UTF-8 first since it's the common
    # case, and only fall back to latin-1 (which accepts every possible
    # byte value, so it can't itself fail to decode) if that raises.
    try:
        raw = pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        raw = pd.read_csv(uploaded_file, encoding='latin-1')

    c = {field: _resolve_column(field, raw.columns) for field in _REQUIRED_FIELDS}

    raw = raw[raw[c['voltage']] > 1.0].reset_index(drop=True)

    # power_gross has no FC_A column at all (computed from voltage * current
    # instead), so it's resolved separately rather than through
    # _resolve_column, which would otherwise raise since FC_A's mapping for
    # it is None.
    gross_col = COLUMN_SCHEMA_VSPY['power_gross']
    if gross_col in raw.columns:
        gross_power = raw[gross_col].values
    else:
        gross_power = raw[c['voltage']].values * raw[c['current']].values

    processed = pd.DataFrame({
        'time'           : raw[c['time']].values,
        'current_density': raw[c['current']].values / Area,
        'bypass_flow'    : (raw[c['bypass_flow']].values * 1.065) / 60.0,
        'wp_rps'         : raw[c['coolant_pump']].values / 60.0,
        'h2_p'           : raw[c['h2_pressure']].values * 1000.0,
        'air_p'          : raw[c['air_pressure']].values * 1000.0,
        'cell_voltage'   : raw[c['voltage']].values,
        'power_request'  : raw[c['power_request']].values,
        'power_net'      : raw[c['power_net']].values,
        'power_gross'    : gross_power,
        # request_slope was previously read from a raw 'Power Difference'
        # column that only exists in the VSpy schema. It's never actually
        # read anywhere downstream (checked run_inference and the ECU
        # feature builders), so it's computed here instead, works
        # identically for both schemas and doesn't need its own raw column.
        'request_slope'  : np.diff(raw[c['power_request']].values,
                                    prepend=raw[c['power_request']].values[0]),
        'T_in_K'         : raw[c['coolant_in_temp']].values + 273.15,
        'T_out_K'        : raw[c['coolant_out_temp']].values + 273.15,
        'air_compressor' : raw[c['air_compressor']].values,
        'H2_consumption' : raw[c['h2_consumption']].values,
    })
    dt      = processed['time'].values[1] - processed['time'].values[0]
    t_start = processed['T_in_K'].values[0]
    return processed, dt, t_start

def build_drive_cycle_from_table(steps, dt=0.1):
    """
    Converts a manual drive cycle table into a minimal inference dataframe.
    steps: list of dicts with keys: type ('constant' or 'ramp'), power_kw, duration_s
    Returns a dataframe with time and power_request columns (others filled with defaults).
    """
    time_arr  = []
    power_arr = []

    t = 0.0
    for step in steps:
        n = int(step['duration_s'] / dt)
        p_start = step.get('power_start_kw', step['power_kw']) * 1000.0
        p_end   = step['power_kw'] * 1000.0

        if step['type'] == 'ramp':
            segment = np.linspace(p_start, p_end, n)
        else:
            segment = np.full(n, p_end)

        times = t + np.arange(n) * dt
        time_arr.append(times)
        power_arr.append(segment)
        t += n * dt

    time_arr  = np.concatenate(time_arr)
    power_arr = np.concatenate(power_arr)

    # Fill in placeholder sensor columns — the ECUs will predict real values
    n_total = len(time_arr)
    df = pd.DataFrame({
        'time'           : time_arr,
        'power_request'  : power_arr,
        'current_density': np.zeros(n_total),   # ECU will fill
        'bypass_flow'    : np.zeros(n_total),   # ECU will fill
        'wp_rps'         : np.zeros(n_total),   # ECU will fill
        'h2_p'           : np.zeros(n_total),   # ECU will fill
        'air_p'          : np.zeros(n_total),   # ECU will fill
        'cell_voltage'   : np.zeros(n_total),
        'power_net'      : np.zeros(n_total),
        'power_gross'    : np.zeros(n_total),
        'request_slope'  : np.diff(power_arr, prepend=power_arr[0]),
        'T_in_K'         : np.zeros(n_total),   # thermal model will fill
        'T_out_K'        : np.zeros(n_total),   # physics will fill
        'air_compressor' : np.zeros(n_total),
        'H2_consumption' : np.zeros(n_total),
    })
    return df, dt

# ── MODEL LOADER ──────────────────────────────────────────────────────────────
def load_all_models(model_dir='.'):
    """Loads all trained .pkl files. Call once at app startup."""
    import os
    def load(fname):
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}\nRun your notebook first to generate all .pkl files.")
        return joblib.load(path)

    models = {
        'SOLVED_PARAMS' : load('solved_params.pkl'),
        'ecu_current'   : load('ecu_current_model.pkl'),
        'h2_regressor'  : load('h2_linear_model.pkl'),
        'ecu_air'       : load('ecu_air_model.pkl'),
        'ecu_bp'        : load('ecu_bypass_model.pkl'),
        'ecu_wp'        : load('ecu_coolant_model.pkl'),
        'svr_model'     : load('svr_single_model.pkl'),
        'scaler_svr'    : load('svr_single_scaler.pkl'),
        'ecu_h2'        : load('ecu_h2_model.pkl'),
        'aux_params'    : load('aux_model_params.pkl'),
        'tin_params'    : load('tin_thermal_model.pkl'),
    }
    return models

# ── FULL INFERENCE PIPELINE ───────────────────────────────────────────────────
def run_inference(df_input, dt, t_start_K, models, lab_rh=0.48):
    """
    Full autonomous inference pipeline.
    Accepts a processed dataframe (from CSV or manual builder).
    Returns a results dict with all predicted arrays.
    """
    SOLVED_PARAMS = models['SOLVED_PARAMS']
    ecu_current   = models['ecu_current']
    h2_regressor  = models['h2_regressor']
    ecu_air       = models['ecu_air']
    ecu_bp        = models['ecu_bp']
    ecu_wp        = models['ecu_wp']
    svr_model     = models['svr_model']
    scaler_svr    = models['scaler_svr']
    ecu_h2        = models['ecu_h2']
    aux_params    = models['aux_params']
    tin_params    = models['tin_params']

    df = df_input.reset_index(drop=True).copy()

    # 1. ECU predictions
    df['current_density'] = ecu_current.predict(build_ecu_features(df))
    df['h2_p']            = h2_regressor.predict(df['power_request'].values.reshape(-1, 1))
    df['air_p']           = ecu_air.predict(build_ecu_features_air(df))
    df['bypass_flow']     = ecu_bp.predict(build_ecu_features_bp(df))
    df['wp_rps']          = ecu_wp.predict(build_ecu_features_wp(df))

    # 2. T_in thermal model
    hg = tin_params['heat_gain']
    cb = tin_params['cooling_base']
    cw = tin_params['cooling_wp']
    cp_val = tin_params['cooling_bp']
    ci = tin_params['cooling_interact']
    tin_pred = predict_tin_actuator_dynamic_nonlinear(
        df, t_start_K, hg, cb, cw, cp_val, ci, fixed_dt=dt)
    df['T_in_K'] = tin_pred

    # 3. Physics loop: voltage + T_out
    V_phys, T_out_phys = predict_voltage_loop(SOLVED_PARAMS, df, dt, t_start_K)
    df['T_out_K'] = T_out_phys

    # 4. SVR correction
    def build_svr_features(df_s, V_physics):
        return np.column_stack([
            df_s['current_density'].values,
            df_s['h2_p'].values,
            df_s['air_p'].values,
            df_s['T_in_K'].values,
            df_s['power_request'].values,
            V_physics,
        ])

    X_svr   = build_svr_features(df, V_phys)
    X_scaled= scaler_svr.transform(X_svr)
    svr_corr= np.clip(svr_model.predict(X_scaled), -10, 10)
    V_final = V_phys + svr_corr

    # 5. T_stack approximation (average of inlet/outlet)
    T_stack = (tin_pred + T_out_phys) / 2.0

    # 6. Gross power = V_final * I_stack
    I_stack   = df['current_density'].values * Area
    p_gross   = V_final * I_stack

    # 7. Auxiliary & net power
    N_ref = aux_params['N_ref']
    P_ref = aux_params['P_ref']
    AUX_SMALL_FRACTION = aux_params['AUX_SMALL_FRACTION']
    p_net, p_comp, p_aux_small, p_converter = compute_net_power(
        p_gross, df['air_p'].values, N_ref, P_ref, AUX_SMALL_FRACTION)
    p_aux_total = p_comp + p_aux_small + p_converter

    # 8. H2 consumption
    faraday   = faraday_h2(df['current_density'].values)
    X_h2      = build_h2_features(df, faraday)
    h2_delta  = ecu_h2.predict(X_h2)
    h2_rate   = faraday + h2_delta   # mg/s
    h2_cumulative = np.cumsum(h2_rate) * dt / 1000.0  # grams

    # ── REAL / MEASURED SIGNAL PASSTHROUGH ────────────────────────────────────
    # df_input is the ORIGINAL uploaded/built dataframe, untouched by the ECU
    # overwrites above (those only mutate the local `df` copy). For a real
    # VehicleSpy CSV, every one of these columns holds actual sensor data.
    # For manual-table / script cycles the columns exist but are all zeros
    # (see build_drive_cycle_from_table) — harmless, since app.py only draws
    # them when st.session_state.has_real_data is True (CSV mode).
    n = len(df)
    def _real(col):
        return df_input[col].values if col in df_input.columns else np.zeros(n)

    return {
        'time'          : df['time'].values,
        'power_request' : df['power_request'].values / 1000.0,  # kW
        'current_density': df['current_density'].values,
        'V_final'       : V_final,
        'V_phys'        : V_phys,
        'T_in_K'        : tin_pred,
        'T_out_K'       : T_out_phys,
        'T_stack_K'     : T_stack,
        'h2_p'          : df['h2_p'].values,
        'air_p'         : df['air_p'].values,
        'bypass_flow'   : df['bypass_flow'].values,
        'wp_rps'        : df['wp_rps'].values,
        'p_gross_kw'    : p_gross / 1000.0,
        'p_net_kw'      : p_net   / 1000.0,
        'p_aux_kw'      : p_aux_total / 1000.0,
        'p_comp_kw'     : p_comp  / 1000.0,
        'p_aux_small_kw': p_aux_small / 1000.0,
        'p_converter_kw': p_converter / 1000.0,
        'h2_rate_mgs'   : h2_rate,
        'h2_cumulative_g': h2_cumulative,
        'svr_correction': svr_corr,
        # ── Every channel below has a measured counterpart when has_real_data
        # is True. Previously only V_real/T_in_real/T_out_real existed, which
        # is why most Panel 3 plots could never show a measured trace — the
        # data simply was never being passed through at all.
        'V_real'              : _real('cell_voltage'),
        'T_in_real'           : _real('T_in_K'),
        'T_out_real'          : _real('T_out_K'),
        'current_density_real': _real('current_density'),
        'h2_p_real'           : _real('h2_p'),
        'air_p_real'          : _real('air_p'),
        'bypass_flow_real'    : _real('bypass_flow'),
        'wp_rps_real'         : _real('wp_rps'),
        'p_net_real_kw'       : _real('power_net')   / 1000.0,
        'p_gross_real_kw'     : _real('power_gross') / 1000.0,
        'h2_rate_real_mgs'    : _real('H2_consumption'),
    }