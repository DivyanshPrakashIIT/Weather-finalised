"""
app/main.py
============================================================
PHASE: DEPLOYMENT — Streamlit Weather Prediction App
Delhi Weather Forecasting – XGBoost + LightGBM Ensemble

  - Loads XGBoost + LightGBM models  (from 04_model_train_evaluate.py)
  - Shows per-model predictions + weighted ensemble
  - Displays historical test results

Run:  streamlit run app/main.py
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Delhi Weather Forecaster",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
    html, body, * { font-family: 'DM Sans', sans-serif !important; }
    .stApp { background: #0D1B2A !important; }
    [data-testid="stAppViewContainer"] { background: #0D1B2A !important; }
    [data-testid="stMain"] { background: #0D1B2A !important; }
    [data-testid="stMainBlockContainer"] { background: #0D1B2A !important; }
    [data-testid="stSidebar"] { background: #112336 !important; border-right: 1px solid #1E3A5F; }
    [data-testid="stSidebarContent"] { background: #112336 !important; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #C8DCEF !important; }
    .app-header {
        background: linear-gradient(135deg, #0D1B2A 0%, #1A3050 50%, #0D1B2A 100%);
        border: 1px solid #1E3A5F; border-radius: 16px;
        padding: 2rem 2.5rem; margin-bottom: 1.5rem;
        position: relative; overflow: hidden;
    }
    .app-header::before {
        content: ''; position: absolute; top: -50%; right: -10%;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(52,152,219,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .app-title { font-size: 2rem; font-weight: 700; color: #E8F4FD; margin: 0; letter-spacing: -0.03em; }
    .app-subtitle { color: #6B9EC7; font-size: 0.95rem; margin-top: 0.3rem; font-weight: 400; }
    .pred-card {
        background: #112336; border: 1px solid #1E3A5F;
        border-radius: 14px; padding: 1.8rem 1.5rem;
        text-align: center; transition: border-color 0.2s, transform 0.15s; height: 100%;
    }
    .pred-card:hover { border-color: #2E6DA4; transform: translateY(-2px); }
    .pred-card.xgb { border-color: #3498DB; background: linear-gradient(160deg, #0A1A2E 0%, #112336 100%); }
    .pred-card.lgb { border-color: #2ECC71; background: linear-gradient(160deg, #0A1E14 0%, #112336 100%); }
    .pred-model-name { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.6rem; }
    .pred-value { font-size: 3rem; font-weight: 700; font-family: 'DM Mono', monospace; line-height: 1; }
    .pred-emoji { font-size: 1rem; color: #8ABADC; margin-top: 0.5rem; }
    .pred-badge {
        display: inline-block; font-size: 0.68rem; font-weight: 600;
        letter-spacing: 0.06em; padding: 0.2rem 0.6rem; border-radius: 20px; margin-top: 0.6rem;
    }
    .badge-xgb { background: #0D2340; color: #3498DB; border: 1px solid #3498DB; }
    .badge-lgb { background: #0D2318; color: #2ECC71; border: 1px solid #2ECC71; }
    .ensemble-hero {
        background: linear-gradient(135deg, #1A1000 0%, #1A2030 100%);
        border: 2px solid #F39C12; border-radius: 16px;
        padding: 2rem 2.5rem; margin-top: 1.5rem;
        display: flex; align-items: center; justify-content: space-between;
        flex-wrap: wrap; gap: 1.5rem;
    }
    .section-header {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: #3A7BBF;
        border-bottom: 1px solid #1E3A5F; padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0;
    }
    .status-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.5rem 0; }
    .status-pill {
        display: inline-flex; align-items: center; gap: 0.35rem;
        font-size: 0.75rem; font-weight: 500; padding: 0.25rem 0.75rem; border-radius: 20px;
    }
    .pill-ok   { background: #0E2318; color: #27AE60; border: 1px solid #1A4028; }
    .pill-miss { background: #1A1522; color: #8E44AD; border: 1px solid #2D1A3D; }
    .pill-warn { background: #1A1000; color: #F39C12; border: 1px solid #3D2800; }
    .info-row { background: #0A1520; border: 1px solid #162840; border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 0.6rem; }
    .info-label { font-size: 0.75rem; font-weight: 600; color: #3A7BBF; }
    .info-value { font-size: 0.9rem; color: #C8DCEF; margin-top: 0.2rem; }
    .metric-table { width: 100%; border-collapse: collapse; }
    .metric-table th { font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; color: #3A7BBF; padding: 0.4rem 0.6rem; border-bottom: 1px solid #1E3A5F; text-align: left; }
    .metric-table td { font-size: 0.85rem; color: #C8DCEF; padding: 0.4rem 0.6rem; border-bottom: 1px solid #0D1B2A; font-family: 'DM Mono', monospace; }
    .metric-table tr:last-child td { border-bottom: none; }
    .metric-table tr.best-row td  { color: #27AE60; }
    [data-testid="stButton"] > button,
    [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, #1A3A6B, #2E6DA4) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
        font-size: 1rem !important; padding: 0.6rem 1.5rem !important;
        width: 100% !important; letter-spacing: 0.02em !important;
    }
    [data-testid="stMarkdownContainer"] p { color: #9BBCDA; }
    h1, h2, h3 { color: #E8F4FD !important; }
    [data-testid="stTabs"] [data-baseweb="tab"] { color: #6B9EC7 !important; background: transparent !important; }
    [data-testid="stTabs"] [aria-selected="true"] { color: #E8F4FD !important; border-bottom: 2px solid #3A7BBF !important; }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { background: #0D1B2A !important; border-bottom: 1px solid #1E3A5F; }
    [data-testid="stTabs"] [data-baseweb="tab-panel"] { background: #0D1B2A !important; }
    [data-testid="stMetric"] { background: #112336 !important; border-radius: 10px !important; padding: 0.8rem !important; border: 1px solid #1E3A5F !important; }
    [data-testid="stMetricLabel"] { color: #6B9EC7 !important; font-size: 0.75rem !important; }
    [data-testid="stMetricLabel"] p { color: #6B9EC7 !important; }
    [data-testid="stMetricValue"] { color: #E8F4FD !important; }
    [data-testid="stMetricValue"] > div { color: #E8F4FD !important; }
    [data-testid="stNumberInput"] input { background: #0A1520 !important; color: #C8DCEF !important; border-color: #1E3A5F !important; }
    [data-testid="stWidgetLabel"] { color: #C8DCEF !important; }
    [data-testid="stWidgetLabel"] p { color: #C8DCEF !important; }
    [data-testid="stDataFrame"] { border: 1px solid #1E3A5F !important; border-radius: 8px !important; }
    [data-testid="stAlert"] { background: #112336 !important; border-color: #1E3A5F !important; }
</style>
""", unsafe_allow_html=True)


def temp_emoji(t):
    if t < 8:   return "🥶 Very Cold"
    if t < 16:  return "🌨️ Cold"
    if t < 22:  return "🌥️ Cool"
    if t < 28:  return "🌤️ Pleasant"
    if t < 34:  return "☀️ Warm"
    return "🔥 Hot"

def temp_color(t):
    if t < 16: return "#74B9FF"
    if t < 26: return "#55EFC4"
    if t < 32: return "#FDCB6E"
    return "#FF7675"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_all_models():
    models = {}
    def try_load(key, path):
        full = os.path.join(BASE_DIR, path)
        if os.path.exists(full):
            try:
                models[key] = joblib.load(full)
            except Exception as e:
                models[key] = None
                models[f'{key}_err'] = str(e)
        else:
            models[key] = None
    try_load('xgb',  'models/xgboost_model.pkl')
    try_load('lgb',  'models/lightgbm_model.pkl')
    try_load('meta', 'models/feature_meta.pkl')
    ens_path = os.path.join(BASE_DIR, 'data/predictions/ensemble_final.csv')
    if os.path.exists(ens_path):
        try:
            models['history'] = pd.read_csv(ens_path, parse_dates=['date'])
        except Exception:
            models['history'] = None
    else:
        models['history'] = None
    return models

models   = load_all_models()
xgb_ok   = models.get('xgb')  is not None
lgb_ok   = models.get('lgb')  is not None
meta_ok  = models.get('meta') is not None
ml_ready = xgb_ok and lgb_ok and meta_ok


def build_feature_row(temp, humidity, wind, pressure, temp_lag1, temp_lag2,
                      temp_lag3=None, temp_lag7=None):
    today  = datetime.date.today()
    month  = today.month
    doy    = today.timetuple().tm_yday
    lag3   = temp_lag3 if temp_lag3 is not None else temp_lag2
    lag7   = temp_lag7 if temp_lag7 is not None else (temp + temp_lag1 + temp_lag2) / 3
    avg3   = (temp_lag1 + temp_lag2 + lag3) / 3
    row = {
        'humidity': humidity, 'wind_speed': wind, 'meanpressure': pressure,
        'month': month, 'day': today.day, 'day_of_year': doy,
        'day_of_week': today.weekday(),
        'season': {12:1,1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4}[month],
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
        'doy_sin': np.sin(2 * np.pi * doy / 365),
        'doy_cos': np.cos(2 * np.pi * doy / 365),
        'temp_lag1': temp_lag1, 'temp_lag2': temp_lag2,
        'temp_lag3': lag3, 'temp_lag7': lag7,
        'humidity_lag1': humidity, 'humidity_lag2': humidity,
        'humidity_lag3': humidity, 'humidity_lag7': humidity,
        'pressure_lag1': pressure, 'pressure_lag2': pressure,
        'pressure_lag3': pressure, 'pressure_lag7': pressure,
        'wind_lag1': wind, 'wind_lag2': wind, 'wind_lag3': wind, 'wind_lag7': wind,
        'temp_roll_mean3': avg3, 'temp_roll_std3': abs(temp_lag1 - temp_lag2) * 0.5,
        'hum_roll_mean3': humidity,
        'temp_roll_mean7': avg3, 'temp_roll_std7': abs(temp_lag1 - temp_lag2) * 0.7,
        'hum_roll_mean7': humidity,
        'temp_roll_mean14': avg3, 'temp_roll_std14': abs(temp_lag1 - temp_lag2) * 0.8,
        'hum_roll_mean14': humidity,
        'temp_ewm7': 0.7 * temp_lag1 + 0.3 * temp_lag2,
        'temp_ewm14': 0.6 * temp_lag1 + 0.4 * temp_lag2,
        'heat_index': temp_lag1 * humidity / 100,
        'pressure_delta': 0.0, 'temp_delta': temp_lag1 - temp_lag2,
        'wind_chill': temp_lag1 - 0.5 * wind,
    }
    features = models['meta']['features']
    return pd.DataFrame([row]).reindex(columns=features, fill_value=0.0)


def run_predictions(temp, humidity, wind, pressure, lag1, lag2):
    preds = {}
    if ml_ready:
        X = build_feature_row(temp, humidity, wind, pressure, lag1, lag2)
        preds['XGBoost']  = float(models['xgb'].predict(X)[0])
        preds['LightGBM'] = float(models['lgb'].predict(X)[0])
    if 'XGBoost' in preds and 'LightGBM' in preds:
        preds['Ensemble']  = preds['XGBoost'] * 0.5 + preds['LightGBM'] * 0.5
        preds['_weights']  = {'XGBoost': 0.50, 'LightGBM': 0.50}
    return preds


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌡️ Today's Observations")
    st.markdown("*Enter current weather readings for Delhi*")
    st.markdown("---")
    temp     = st.slider("Mean Temperature (°C)", 0.0, 45.0, 24.0, 0.5)
    humidity = st.slider("Humidity (%)", 10.0, 100.0, 62.0, 1.0)
    wind     = st.slider("Wind Speed (km/h)", 0.0, 50.0, 8.0, 0.5)
    pressure = st.slider("Mean Pressure (hPa)", 990.0, 1030.0, 1010.0, 0.5)
    st.markdown("---")
    st.markdown("##### 📅 Recent Temperature History")
    lag1 = st.number_input("Yesterday (°C)",  value=23.0, step=0.5)
    lag2 = st.number_input("2 Days Ago (°C)", value=22.0, step=0.5)
    st.markdown("---")
    predict_btn = st.button("🔮 Predict Tomorrow's Temperature")
    st.markdown("---")
    st.markdown("##### ⚙️ Model Status")
    status_html = '<div class="status-row">'
    for name, ok in [('XGBoost', xgb_ok), ('LightGBM', lgb_ok)]:
        cls  = 'pill-ok' if ok else 'pill-miss'
        icon = '✓' if ok else '✗'
        status_html += f'<span class="status-pill {cls}">{icon} {name}</span>'
    status_html += '</div>'
    st.markdown(status_html, unsafe_allow_html=True)
    if not ml_ready:
        st.markdown(
            '<div class="status-pill pill-warn" style="margin-top:0.5rem">'
            '⚠ Run Steps 6–7 in Colab to train XGB/LGB</div>',
            unsafe_allow_html=True)


# ── MAIN PANEL ───────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="app-title">🌡️ Delhi Weather Forecaster</div>
  <div class="app-subtitle">
    XGBoost &amp; LightGBM · Equal-weight Ensemble ·
    Trained on Delhi 2013–2017 daily observations
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Historical Results", "ℹ️ Model Info"])


# ── TAB 1: PREDICT ───────────────────────────────────────────
with tab1:
    if not predict_btn:
        st.markdown("""
        <div style="background:#112336;border:1px dashed #1E3A5F;border-radius:14px;
                    padding:3rem 2rem;text-align:center;margin-top:1rem;">
          <div style="font-size:3rem;margin-bottom:1rem;">🌤️</div>
          <div style="color:#6B9EC7;font-size:1.1rem;font-weight:500;">
            Set today's weather values in the sidebar,<br>
            then press <b style="color:#E8F4FD">Predict Tomorrow's Temperature</b>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if not ml_ready:
            st.error("⚠️ XGBoost/LightGBM models not found. Run Steps 6–7 in the Colab notebook first.")
        else:
            preds   = run_predictions(temp, humidity, wind, pressure, lag1, lag2)
            weights = preds.pop('_weights', {})
            ensemble_val = preds.get('Ensemble')

            st.markdown('<div class="section-header">Individual Model Forecasts</div>',
                        unsafe_allow_html=True)
            col_xgb, col_lgb = st.columns(2)
            xgb_val = preds.get('XGBoost', 0)
            lgb_val = preds.get('LightGBM', 0)

            with col_xgb:
                st.markdown(f"""
                <div class="pred-card xgb">
                  <div class="pred-model-name" style="color:#3498DB;">⚡ XGBoost</div>
                  <div class="pred-value" style="color:{temp_color(xgb_val)};">{xgb_val:.1f}°C</div>
                  <div class="pred-emoji">{temp_emoji(xgb_val)}</div>
                  <span class="pred-badge badge-xgb">500 Trees · Gradient Boosting</span>
                </div>""", unsafe_allow_html=True)

            with col_lgb:
                st.markdown(f"""
                <div class="pred-card lgb">
                  <div class="pred-model-name" style="color:#2ECC71;">🌿 LightGBM</div>
                  <div class="pred-value" style="color:{temp_color(lgb_val)};">{lgb_val:.1f}°C</div>
                  <div class="pred-emoji">{temp_emoji(lgb_val)}</div>
                  <span class="pred-badge badge-lgb">Leaf-wise · Histogram Boosting</span>
                </div>""", unsafe_allow_html=True)

            if ensemble_val is not None:
                diff       = ensemble_val - temp
                diff_sign  = "+" if diff >= 0 else ""
                diff_color = "#FF7675" if diff > 0 else "#74B9FF" if diff < 0 else "#C8DCEF"
                st.markdown(f"""
                <div class="ensemble-hero">
                  <div>
                    <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.1em;
                        text-transform:uppercase;color:#F39C12;margin-bottom:0.4rem;">
                        ★ Final Ensemble Prediction</div>
                    <div style="font-family:'DM Mono',monospace;font-size:3.8rem;
                        font-weight:700;color:{temp_color(ensemble_val)};line-height:1;">
                        {ensemble_val:.1f}°C</div>
                    <div style="font-size:1rem;color:#8ABADC;margin-top:0.4rem;">
                        {temp_emoji(ensemble_val)}</div>
                  </div>
                  <div style="text-align:right;">
                    <div style="font-size:0.68rem;color:#5B8AB0;font-weight:700;
                        letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.6rem;">
                        Weight Distribution</div>
                    <div style="font-size:0.9rem;color:#C8DCEF;
                        font-family:'DM Mono',monospace;line-height:2.2;">
                        XGBoost: <b>50%</b><br>LightGBM: <b>50%</b></div>
                    <div style="margin-top:0.8rem;font-size:0.85rem;
                        color:{diff_color};font-family:'DM Mono',monospace;">
                        vs Today: <b>{diff_sign}{diff:.1f}°C</b></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header">Today\'s Conditions</div>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Today's Temp",  f"{temp:.1f}°C")
            c2.metric("Yesterday",      f"{lag1:.1f}°C", delta=f"{temp-lag1:+.1f}°C")
            c3.metric("Humidity",       f"{humidity:.0f}%")
            c4.metric("Pressure",       f"{pressure:.0f} hPa")

            st.markdown('<div class="section-header">Model Comparison Chart</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(8, 3.5))
            fig.patch.set_facecolor('#0D1B2A')
            ax.set_facecolor('#112336')
            plot_models = ['XGBoost', 'LightGBM', 'Ensemble']
            vals   = [preds[m] for m in plot_models]
            colors = ['#3498DB', '#2ECC71', '#F39C12']
            bars   = ax.bar(plot_models, vals, color=colors, edgecolor='#1E3A5F',
                            linewidth=0.8, width=0.45)
            y_min = max(0, min(vals) - 4)
            y_max = max(vals) + 4
            ax.set_ylim(y_min, y_max)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.1f}°C', ha='center', va='bottom',
                        color='#C8DCEF', fontsize=11, fontweight='700')
            ax.axhline(ensemble_val, color='#F39C12', lw=1.4, ls='--', alpha=0.45)
            ax.set_ylabel('Temperature (°C)', color='#6B9EC7', fontsize=10)
            ax.tick_params(colors='#6B9EC7', labelsize=10)
            for spine in ax.spines.values():
                spine.set_edgecolor('#1E3A5F')
            ax.set_title("Tomorrow's Temperature — XGBoost vs LightGBM vs Ensemble",
                         color='#9BBCDA', fontsize=10, pad=10)
            from matplotlib.patches import Patch
            ax.legend(handles=[
                Patch(facecolor='#3498DB', label='XGBoost'),
                Patch(facecolor='#2ECC71', label='LightGBM'),
                Patch(facecolor='#F39C12', label='Ensemble (50/50)'),
            ], fontsize=9, facecolor='#0D1B2A', labelcolor='#9BBCDA',
               edgecolor='#1E3A5F', loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ── TAB 2: HISTORICAL ────────────────────────────────────────
with tab2:
    history = models.get('history')
    if history is None:
        st.info("Historical predictions not found. Run Steps 7–10 in the Colab notebook to generate `data/predictions/ensemble_final.csv`.")
    else:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        actual   = history['actual'].values
        xgb_col  = next((c for c in history.columns if 'xgb' in c.lower()), None)
        lgb_col  = next((c for c in history.columns if 'lgb' in c.lower() or 'light' in c.lower()), None)
        ens_col  = next((c for c in history.columns if 'ensemble' in c.lower()), None)
        pred_cols = [c for c in [xgb_col, lgb_col, ens_col] if c is not None]
        if not pred_cols:
            pred_cols = [c for c in history.columns if c.startswith('pred_') or c == 'prediction_ensemble']
        model_labels = {c: c.replace('pred_','').replace('_',' ').upper() for c in pred_cols}

        st.markdown('<div class="section-header">Test Set Performance (Delhi 2017)</div>', unsafe_allow_html=True)
        rows = []
        for col in pred_cols:
            p    = history[col].values
            rmse = np.sqrt(mean_squared_error(actual, p))
            mae  = mean_absolute_error(actual, p)
            mape = np.mean(np.abs((actual - p) / (np.abs(actual) + 1e-8))) * 100
            r2   = r2_score(actual, p)
            rows.append({'Model': model_labels[col], 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2})
        metrics_df = pd.DataFrame(rows).sort_values('RMSE').reset_index(drop=True)
        best_rmse  = metrics_df['RMSE'].min()
        table_html = '<table class="metric-table"><tr>'
        for col in ['Model', 'RMSE', 'MAE', 'MAPE', 'R²']:
            table_html += f'<th>{col}</th>'
        table_html += '</tr>'
        for _, row in metrics_df.iterrows():
            is_best = row['RMSE'] == best_rmse
            cls  = 'class="best-row"' if is_best else ''
            star = ' ★' if is_best else ''
            table_html += f'<tr {cls}><td>{row["Model"]}{star}</td><td>{row["RMSE"]:.4f} °C</td><td>{row["MAE"]:.4f} °C</td><td>{row["MAPE"]:.2f}%</td><td>{row["R²"]:.4f}</td></tr>'
        table_html += '</table>'
        st.markdown(f'<div style="background:#112336;border:1px solid #1E3A5F;border-radius:12px;padding:1rem;">{table_html}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Actual vs Predicted — XGBoost &amp; LightGBM</div>', unsafe_allow_html=True)
        dates   = pd.to_datetime(history['date']).values
        palette = ['#3498DB', '#2ECC71', '#F39C12']
        n_plots = len(pred_cols)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.5 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]
        fig.patch.set_facecolor('#0D1B2A')
        for ax, col, color in zip(axes, pred_cols, palette):
            pred = history[col].values
            row  = metrics_df[metrics_df['Model'] == model_labels[col]].iloc[0]
            ax.set_facecolor('#112336')
            ax.plot(dates, actual, color='#E8F4FD', lw=1.6, label='Actual', zorder=3)
            ax.plot(dates, pred,   color=color, lw=1.4, ls='--', label=model_labels[col], zorder=2)
            ax.fill_between(dates, actual, pred, alpha=0.08, color=color)
            ax.set_title(f'{model_labels[col]}  ·  RMSE={row["RMSE"]:.3f}°C  MAE={row["MAE"]:.3f}°C  MAPE={row["MAPE"]:.2f}%  R²={row["R²"]:.4f}',
                         color='#9BBCDA', fontsize=9, fontweight='600', pad=6)
            ax.set_ylabel('°C', color='#6B9EC7', fontsize=9)
            ax.tick_params(colors='#6B9EC7', labelsize=8)
            ax.legend(fontsize=8, facecolor='#0D1B2A', labelcolor='#9BBCDA', edgecolor='#1E3A5F')
            ax.grid(True, alpha=0.15, color='#1E3A5F')
            for spine in ax.spines.values():
                spine.set_edgecolor('#1E3A5F')
        axes[-1].set_xlabel('Date', color='#6B9EC7', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        ref_col = ens_col or (xgb_col if xgb_col else (pred_cols[0] if pred_cols else None))
        if ref_col and ref_col in history.columns:
            st.markdown('<div class="section-header">Residual Analysis</div>', unsafe_allow_html=True)
            residuals = actual - history[ref_col].values
            fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
            fig.patch.set_facecolor('#0D1B2A')
            for ax in axes:
                ax.set_facecolor('#112336')
                for spine in ax.spines.values(): spine.set_edgecolor('#1E3A5F')
                ax.tick_params(colors='#6B9EC7', labelsize=8)
            axes[0].plot(dates, residuals, color='#F39C12', lw=1.2)
            axes[0].axhline(0, color='#E74C3C', lw=1.3, ls='--', alpha=0.7)
            axes[0].fill_between(dates, residuals, 0, alpha=0.12, color='#F39C12')
            axes[0].set_title('Residuals Over Time', color='#9BBCDA', fontsize=9)
            axes[0].set_ylabel('Error (°C)', color='#6B9EC7', fontsize=9)
            axes[0].grid(True, alpha=0.15, color='#1E3A5F')
            axes[1].hist(residuals, bins=22, color='#3A7BBF', alpha=0.85, edgecolor='#1E3A5F')
            axes[1].axvline(0, color='#E74C3C', lw=1.3, ls='--', alpha=0.7, label='Zero error')
            axes[1].axvline(residuals.mean(), color='#F39C12', lw=1.3, ls='--',
                            label=f'Mean = {residuals.mean():.2f}°C')
            axes[1].set_title('Residual Distribution', color='#9BBCDA', fontsize=9)
            axes[1].set_xlabel('Error (°C)', color='#6B9EC7', fontsize=9)
            axes[1].legend(fontsize=8, facecolor='#0D1B2A', labelcolor='#9BBCDA', edgecolor='#1E3A5F')
            axes[1].grid(True, alpha=0.15, color='#1E3A5F')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ── TAB 3: MODEL INFO ────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Architecture Overview</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        for title, body in [
            ("⚡ XGBoost",
             "500 trees · max_depth=5 · learning_rate=0.05 · early stopping\n"
             "Features: 43 engineered (lags, rolling stats, cyclical, interactions)\n"
             "Regularization: L1 + L2 · Subsample: 0.8"),
            ("🌿 LightGBM",
             "500 trees · num_leaves=31 · learning_rate=0.05 · early stopping\n"
             "Features: same 43-feature set as XGBoost\n"
             "Histogram-based · Leaf-wise growth · Faster training"),
        ]:
            st.markdown(f'<div class="info-row"><div class="info-label">{title}</div>'
                        f'<div class="info-value" style="white-space:pre-line">{body}</div></div>',
                        unsafe_allow_html=True)
    with c2:
        for title, body in [
            ("★ Ensemble Strategy",
             "Equal-weight average: XGBoost 50% + LightGBM 50%\n"
             "Both models use identical 43-feature engineering.\n"
             "Ensemble reduces individual model variance while\n"
             "preserving the strong lag/rolling feature signal."),
            ("📦 Dataset",
             "Delhi daily weather 2013–2017\n"
             "Train: 1,462 rows · Test: 114 rows\n"
             "Features: temperature, humidity, wind_speed, pressure"),
        ]:
            st.markdown(f'<div class="info-row"><div class="info-label">{title}</div>'
                        f'<div class="info-value" style="white-space:pre-line">{body}</div></div>',
                        unsafe_allow_html=True)

    st.markdown('<div class="section-header">Feature Engineering Summary (43 Features)</div>',
                unsafe_allow_html=True)
    if meta_ok:
        features = models['meta']['features']
        categories = {
            'Lag Features (temp/hum/pressure/wind)':
                [f for f in features if 'lag' in f],
            'Rolling Stats (mean/std/ewm)':
                [f for f in features if 'roll' in f or 'ewm' in f],
            'Time & Cyclical':
                [f for f in features if any(x in f for x in ['month','day','doy','season','year'])],
            'Interaction Features':
                [f for f in features if any(x in f for x in ['heat','delta','chill','index'])],
            'Raw Weather':
                [f for f in features if f in ['humidity','wind_speed','meanpressure']],
        }
        fc1, fc2 = st.columns(2)
        for i, (cat, feats) in enumerate(categories.items()):
            col = fc1 if i % 2 == 0 else fc2
            with col:
                st.markdown(f'<div class="info-row"><div class="info-label">{cat} ({len(feats)})</div>'
                            f'<div class="info-value" style="font-size:0.78rem;color:#5B8AB0;">'
                            f'{", ".join(feats[:8])}{"..." if len(feats)>8 else ""}</div></div>',
                            unsafe_allow_html=True)
    else:
        st.info("feature_meta.pkl not found — run Steps 6–7 in Colab to generate it.")

    st.markdown('<div class="section-header">Model Registry</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        'Model':     ['XGBoost', 'LightGBM'],
        'Weight':    ['0.500',   '0.500'],
        'Algorithm': ['Gradient Boosted Trees (depth-wise)',
                      'Gradient Boosted Trees (leaf-wise, histogram)'],
        'Status':    ['✓ Loaded' if xgb_ok else '✗ Missing',
                      '✓ Loaded' if lgb_ok else '✗ Missing'],
    }), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="info-row" style="margin-top:0.8rem;">
      <div class="info-label">💡 Why XGBoost + LightGBM?</div>
      <div class="info-value">Both models excel at tabular regression with engineered features.
      XGBoost (depth-wise) and LightGBM (leaf-wise) have complementary inductive biases —
      their ensemble reduces variance without sacrificing the strong temperature-tracking signal
      from the 7-day lag and rolling-mean features.</div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;color:#2A4A6B;font-size:0.78rem;
    margin-top:2rem;padding:1rem 0;border-top:1px solid #1A2E44;">
  Delhi Weather Forecaster · XGBoost + LightGBM Ensemble ·
  Dataset: Delhi 2013–2017 · Author: Divyansh Prakash &amp; Siddharth
</div>
""", unsafe_allow_html=True)
