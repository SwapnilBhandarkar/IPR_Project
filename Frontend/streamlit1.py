import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time

# ─── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="TSLab · Sensor Forecasting Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session state init ─────────────────────────────────────────
for key, default in [
    ("dark_mode",     True),
    ("cache",         {}),
    ("groq_key",    ""),
    ("groq_result", {}),   # {model_key: summary_text}
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Theme tokens ───────────────────────────────────────────────
if st.session_state.dark_mode:
    BG, BG_CARD, BG_PANEL     = "#0d0f14", "#181c26", "#12151c"
    BORDER, BORDER_L           = "#252a38", "#2e3448"
    TEXT, TEXT_MUTED, TEXT_DIM = "#e8ecf4", "#6b7491", "#3e4560"
    GRID, PLOT_BG              = "#1a1e2c", "#12151c"
else:
    BG, BG_CARD, BG_PANEL     = "#f0f2f6", "#ffffff", "#f8f9fb"
    BORDER, BORDER_L           = "#e0e4ef", "#c8cedf"
    TEXT, TEXT_MUTED, TEXT_DIM = "#0d1117", "#5a6484", "#9aa3bf"
    GRID, PLOT_BG              = "#e8ecf4", "#f8f9fb"

PAPER_BG      = "rgba(0,0,0,0)"
ACCENT_BLUE   = "#4f8ef7"
ACCENT_ORANGE = "#ff6b35"
ACCENT_GREEN  = "#00d4aa"
ACCENT_PINK   = "#ff4eb8"
ACCENT_YELLOW = "#ffd93d"
ACCENT_PURPLE = "#a78bfa"

MODELS = ["arima", "sarima", "prophet", "lstm", "rf"]
MODEL_COLORS = {
    "arima":   ACCENT_BLUE,
    "sarima":  ACCENT_ORANGE,
    "prophet": ACCENT_GREEN,
    "lstm":    ACCENT_PINK,
    "rf":      ACCENT_YELLOW,
}
MODEL_LABELS = {
    "arima":   "ARIMA",
    "sarima":  "SARIMA",
    "prophet": "Prophet",
    "lstm":    "LSTM",
    "rf":      "Random Forest",
}

# ─── CSS ────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  .stApp {{ background-color: {BG}; }}
  section[data-testid="stSidebar"] {{ background-color:{BG_PANEL}; border-right:1px solid {BORDER}; }}
  h1,h2,h3,h4 {{ color:{TEXT} !important; }}
  p, label, .stMarkdown p {{ color:{TEXT_MUTED}; }}
  div[data-testid="stMetric"] {{ background:{BG_CARD}; border:1px solid {BORDER}; border-radius:12px; padding:18px 20px; }}
  div[data-testid="stMetric"] label {{ color:{TEXT_MUTED} !important; font-size:11px !important; font-family:monospace !important; letter-spacing:1px; text-transform:uppercase; }}
  div[data-testid="stMetricValue"] div {{ color:{ACCENT_BLUE} !important; font-family:monospace !important; font-size:24px !important; }}
  .stButton>button {{ border-radius:8px !important; font-weight:600 !important; border:none !important; }}
  .stSelectbox>div>div {{ background:{BG_CARD} !important; border-color:{BORDER_L} !important; color:{TEXT} !important; border-radius:8px !important; }}
  .stTabs [data-baseweb="tab-list"] {{ background:{BG_PANEL}; border-radius:10px; padding:4px; gap:4px; border:1px solid {BORDER}; width:100%; }}
  .stTabs [data-baseweb="tab"] {{ flex:1; justify-content:center; background:transparent !important; color:{TEXT_MUTED} !important; border-radius:7px !important; font-family:monospace !important; font-size:12px !important; }}
  .stTabs [aria-selected="true"] {{ background:{BG_CARD} !important; color:{TEXT} !important; border:1px solid {BORDER_L} !important; }}
  .stSuccess {{ background:rgba(0,212,170,.1) !important; border-color:{ACCENT_GREEN} !important; color:{ACCENT_GREEN} !important; }}
  .stError   {{ background:rgba(255,78,184,.1) !important; border-color:{ACCENT_PINK} !important; }}
  .stInfo    {{ background:rgba(79,142,247,.1) !important; border-color:{ACCENT_BLUE} !important; }}
  textarea {{ background:{BG_CARD} !important; color:{TEXT} !important; border-color:{BORDER_L} !important; font-family:monospace !important; font-size:13px !important; }}
  ::-webkit-scrollbar {{ width:5px }} ::-webkit-scrollbar-thumb {{ background:{BORDER_L}; border-radius:3px }}
</style>""", unsafe_allow_html=True)

# ─── Plotly base layout ─────────────────────────────────────────
def plot_layout(**kw):
    base = dict(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_MUTED, family="monospace", size=11),
        margin=dict(l=55, r=25, t=45, b=45),
        xaxis=dict(gridcolor=GRID, showgrid=True, zeroline=False,
                   tickfont=dict(color=TEXT_MUTED), linecolor=BORDER),
        yaxis=dict(gridcolor=GRID, showgrid=True, zeroline=False,
                   tickfont=dict(color=TEXT_MUTED), linecolor=BORDER),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MUTED),
                    bordercolor=BORDER, borderwidth=1),
        hoverlabel=dict(bgcolor=BG_CARD, bordercolor=BORDER_L,
                        font=dict(color=TEXT, family="monospace", size=12)),
    )
    base.update(kw)
    return base

# ─── API helpers ────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

def api_get(path, params=None, timeout=90):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to {API_BASE}. Is uvicorn running?"
    except requests.exceptions.Timeout:
        return None, f"Request timed out after {timeout}s."
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP {e.response.status_code}: {e.response.text[:200]}"
    except Exception as e:
        return None, str(e)

def parse_train(res):
    if not isinstance(res, dict):
        return [], [], {}, [], [], []
    actual       = res.get("actual", [])
    predicted    = res.get("predicted", [])
    metrics_data = res.get("metrics") or {k: res[k] for k in ("mae","rmse","mape") if k in res}
    dates        = res.get("index", [])
    train_actual = res.get("train_actual", [])
    train_index  = res.get("train_index", [])
    # Inject accuracy if not already present
    mape_v = metrics_data.get("mape")
    if mape_v is not None and "accuracy" not in metrics_data:
        metrics_data["accuracy"] = max(0.0, min(100.0, 100.0 - float(mape_v)))
    return actual, predicted, metrics_data, dates, train_actual, train_index

def parse_forecast(res):
    if isinstance(res, list):
        return res, []
    if isinstance(res, dict):
        dates = res.get("index", res.get("dates", []))
        for k in ("forecast","predictions","future_predictions","data","yhat","values"):
            v = res.get(k)
            if isinstance(v, list) and v:
                return v, dates
    return [], []

def get_cache(m):        return st.session_state.cache.get(m, {})
def set_cache(m, k, v):  st.session_state.cache.setdefault(m, {})[k] = v

# ─── Chart helpers ──────────────────────────────────────────────
def chart_actpred(actual, predicted, model, dates=None,
                  train_act=None, train_idx=None):
    fig = go.Figure()
    if train_act and train_idx:
        fig.add_trace(go.Scatter(x=train_idx, y=train_act,
            name="Training Data", mode="lines",
            line=dict(color=ACCENT_BLUE, width=1.5), opacity=0.85))
    x_test = (dates if dates and len(dates) == len(actual)
               else list(range(len(train_act or []),
                               len(train_act or []) + len(actual))))
    if actual:
        fig.add_trace(go.Scatter(x=x_test, y=actual,
            name="Actual Test Data", mode="lines",
            line=dict(color=ACCENT_ORANGE, width=1.5)))
    if predicted:
        fig.add_trace(go.Scatter(x=x_test, y=predicted,
            name="Forecasted Data", mode="lines",
            line=dict(color=ACCENT_GREEN, width=2)))
    fig.update_layout(
        title=dict(text=f"{MODEL_LABELS.get(model, model.upper())} · Forecast vs Actual",
                   font=dict(color=TEXT, size=14, family="monospace")),
        xaxis_title="Datetime" if dates else "Index",
        yaxis_title="Impurity Count",
        hovermode="x unified",
        **plot_layout(height=400))
    return fig

def chart_future(forecast_vals, model, steps, dates=None):
    color = MODEL_COLORS.get(model, ACCENT_GREEN)
    x     = (dates if dates and len(dates) == len(forecast_vals)
              else list(range(1, len(forecast_vals) + 1)))
    lower = [v * 0.97 for v in forecast_vals]
    upper = [v * 1.03 for v in forecast_vals]
    h_     = color.lstrip("#")
    r_, g_, b_ = (int(h_[i:i+2], 16) for i in (0, 2, 4))
    rgba  = f"rgba({r_},{g_},{b_},0.12)"
    fig   = go.Figure()
    fig.add_trace(go.Scatter(x=x+x[::-1], y=upper+lower[::-1],
        fill="toself", fillcolor=rgba,
        line=dict(color="rgba(0,0,0,0)"),
        name="±3 % band", showlegend=True))
    fig.add_trace(go.Scatter(x=x, y=forecast_vals,
        name=f"{MODEL_LABELS.get(model, model.upper())} Forecast",
        mode="lines+markers",
        line=dict(color=color, width=3),
        marker=dict(size=8, color=color, line=dict(color=BG_CARD, width=2))))
    fig.update_layout(
        title=dict(text=f"{MODEL_LABELS.get(model, model.upper())}  ·  {steps}-Step Forecast",
                   font=dict(color=TEXT, size=14, family="monospace")),
        xaxis_title="Date" if dates else "Step",
        yaxis_title="Impurity Count (forecast)",
        **plot_layout(height=370))
    return fig

def chart_residuals(actual, predicted, model, dates=None):
    n     = min(len(actual), len(predicted))
    res   = [actual[i] - predicted[i] for i in range(n)]
    mean_r = sum(res) / len(res)
    x     = dates[:n] if dates and len(dates) >= n else list(range(n))
    bars  = [ACCENT_ORANGE if v > 0 else ACCENT_BLUE for v in res]
    fig   = go.Figure()
    fig.add_trace(go.Bar(x=x, y=res, name="Residual",
                         marker_color=bars, opacity=0.8))
    fig.add_hline(y=0,      line_dash="dash", line_color=BORDER_L)
    fig.add_hline(y=mean_r, line_dash="dot",  line_color=ACCENT_ORANGE,
        annotation_text=f"mean={mean_r:.3f}",
        annotation_font_color=ACCENT_ORANGE)
    fig.update_layout(
        title=dict(text=f"{MODEL_LABELS.get(model, model.upper())}  ·  Residuals (Actual − Predicted)",
                   font=dict(color=TEXT, size=14, family="monospace")),
        xaxis_title="Date" if dates else "Index",
        yaxis_title="Error",
        **plot_layout(height=320))
    return fig

def chart_compare_bar(m1, met1, m2, met2):
    keys   = [k for k in ("mae", "rmse", "mape") if k in met1 or k in met2]
    labels = [k.upper() for k in keys]
    c1, c2 = MODEL_COLORS.get(m1, ACCENT_BLUE), MODEL_COLORS.get(m2, ACCENT_GREEN)
    v1 = [met1.get(k, 0) or 0 for k in keys]
    v2 = [met2.get(k, 0) or 0 for k in keys]
    fig = go.Figure()
    fig.add_trace(go.Bar(name=MODEL_LABELS.get(m1, m1.upper()), x=labels, y=v1,
        marker_color=c1, opacity=0.85,
        text=[f"{v:.2f}" for v in v1], textposition="outside",
        textfont=dict(color=c1, size=11)))
    fig.add_trace(go.Bar(name=MODEL_LABELS.get(m2, m2.upper()), x=labels, y=v2,
        marker_color=c2, opacity=0.85,
        text=[f"{v:.2f}" for v in v2], textposition="outside",
        textfont=dict(color=c2, size=11)))
    fig.update_layout(barmode="group",
        title=dict(text="Model Comparison  ·  Performance Metrics",
                   font=dict(color=TEXT, size=14, family="monospace")),
        **plot_layout(height=380))
    return fig

def chart_overlay(m1, d1, m2, d2):
    fig = go.Figure()
    for m, d in [(m1, d1), (m2, d2)]:
        c      = MODEL_COLORS.get(m, ACCENT_BLUE)
        m_dates = d.get("dates")
        if d.get("actual"):
            x_a = (m_dates if m_dates and len(m_dates) == len(d["actual"])
                   else list(range(len(d["actual"]))))
            fig.add_trace(go.Scatter(x=x_a, y=d["actual"],
                name=f"{MODEL_LABELS.get(m, m.upper())} Actual",
                line=dict(color=c, width=1.5, dash="dot"), opacity=0.5))
        if d.get("predicted"):
            x_p = (m_dates if m_dates and len(m_dates) == len(d["predicted"])
                   else list(range(len(d["predicted"]))))
            fig.add_trace(go.Scatter(x=x_p, y=d["predicted"],
                name=f"{MODEL_LABELS.get(m, m.upper())} Predicted",
                line=dict(color=c, width=2.5), marker=dict(size=4)))
    fig.update_layout(
        title=dict(text="Predicted Values Overlay",
                   font=dict(color=TEXT, size=14, family="monospace")),
        xaxis_title="Date" if d1.get("dates") or d2.get("dates") else "Index",
        yaxis_title="Impurity Count",
        **plot_layout(height=360))
    return fig

# ─── HTML card helpers ──────────────────────────────────────────
def mcard(label, value, desc, color, icon="◈"):
    return f"""<div style="background:{BG_CARD};border:1px solid {BORDER};border-left:3px solid {color};
        border-radius:10px;padding:16px 18px">
      <div style="display:flex;align-items:center;gap:7px;margin-bottom:8px">
        <span style="color:{color};font-size:14px">{icon}</span>
        <span style="font-family:monospace;font-size:10px;color:{TEXT_MUTED};letter-spacing:1px;text-transform:uppercase">{label}</span>
      </div>
      <div style="font-family:monospace;font-size:26px;font-weight:600;color:{TEXT};letter-spacing:-1px">{value}</div>
      <div style="font-family:monospace;font-size:10px;color:{TEXT_DIM};margin-top:5px">{desc}</div>
    </div>"""

def summary_card(name, met, color, is_best=False):
    border = f"2px solid {color}" if is_best else f"1px solid {BORDER}"
    badge  = (
        f"<span style='background:{color}20;color:{color};border:1px solid {color}40;"
        f"border-radius:20px;padding:2px 10px;font-size:10px;font-family:monospace;margin-left:8px'>◆ Best</span>"
        if is_best else ""
    )
    display = []
    if "mae"      in met: display.append(("MAE",      f"{met['mae']:.4f}"))
    if "rmse"     in met: display.append(("RMSE",     f"{met['rmse']:.4f}"))
    if "mape"     in met: display.append(("MAPE",     f"{met['mape']:.2f}%"))
    if "accuracy" in met: display.append(("Accuracy", f"{met['accuracy']:.2f}%"))
    rows = "".join(
        f"<div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid {BORDER}'>"
        f"<span style='color:{TEXT_MUTED};font-family:monospace;font-size:12px'>{lbl}:</span>"
        f"<span style='color:{TEXT};font-family:monospace;font-size:12px;font-weight:600'>{val}</span></div>"
        for lbl, val in display
    )
    return (
        f"<div style='background:{BG_CARD};border:{border};border-radius:12px;padding:18px 20px'>"
        f"<div style='display:flex;align-items:center;margin-bottom:14px'>"
        f"<span style='width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;margin-right:8px'></span>"
        f"<span style='font-size:16px;font-weight:700;color:{TEXT};font-family:monospace'>{MODEL_LABELS.get(name, name.upper())}</span>{badge}</div>"
        f"{rows}</div>"
    )


from groq_summary import get_gemini_summary

def call_groq(api_key: str, model_name: str, metrics_dict: dict,
                actual: list, forecast_vals: list,
                trend_label: str, trend_pct: float) -> str:
    """Delegates to get_gemini_summary() in groq_summary.py."""
    text, err = get_gemini_summary(
        model_name     = model_name,
        actual         = actual,
        predicted      = [],        # residuals computed inside groq_summary
        metrics        = metrics_dict,
        forecast       = forecast_vals,
        forecast_steps = len(forecast_vals) if forecast_vals else 0,
        api_key        = api_key,
    )
    if err:
        raise RuntimeError(err)
    return text


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    hcol, tcol = st.columns([3, 1])
    with hcol:
        st.markdown(f"<h2 style='color:{TEXT};margin:0;font-size:17px'>📈 TSLab</h2>",
                    unsafe_allow_html=True)
    with tcol:
        if st.button("☀️" if st.session_state.dark_mode else "🌙",
                     help="Toggle dark / light mode", key="theme_btn"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown(f"<hr style='border-color:{BORDER};margin:10px 0'>", unsafe_allow_html=True)

    st.markdown(
        f"<p style='font-family:monospace;font-size:10px;color:{TEXT_MUTED};"
        f"letter-spacing:1px;text-transform:uppercase;margin-bottom:6px'>Active Model</p>",
        unsafe_allow_html=True)
    sel_model = st.selectbox(
        "", MODELS,
        format_func=lambda x: MODEL_LABELS[x],
        label_visibility="collapsed", key="model_select")
    mc = MODEL_COLORS[sel_model]
    st.markdown(
        f"<div style='display:inline-flex;align-items:center;gap:6px;background:{mc}15;"
        f"border:1px solid {mc}30;border-radius:20px;padding:3px 10px;margin-bottom:8px'>"
        f"<span style='width:6px;height:6px;border-radius:50%;background:{mc}'></span>"
        f"<span style='font-family:monospace;font-size:10px;color:{mc}'>"
        f"{MODEL_LABELS[sel_model]}</span></div>",
        unsafe_allow_html=True)

    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)

    st.markdown(
        f"<p style='font-family:monospace;font-size:10px;color:{TEXT_MUTED};"
        f"letter-spacing:1px;text-transform:uppercase;margin-bottom:6px'>Forecast Steps</p>",
        unsafe_allow_html=True)
    steps = st.select_slider("", options=[12, 24, 48, 72, 168], value=24,
                              label_visibility="collapsed")

    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)

    # Training time hint for slow models
    slow_models = {"lstm": "~60 s (neural net)", "rf": "~20 s (ensemble)"}
    hint = slow_models.get(sel_model, "")
    spinner_msg = (f"Training {MODEL_LABELS[sel_model]} {hint}…"
                   if hint else f"Training {MODEL_LABELS[sel_model]}…")

    if st.button("⚙  Train Model", use_container_width=True):
        with st.spinner(spinner_msg):
            res, err = api_get(f"/train/{sel_model}", timeout=600)
        if err:
            st.error(err)
        else:
            a, p, m, d, ta, ti = parse_train(res)
            set_cache(sel_model, "actual",       a)
            set_cache(sel_model, "predicted",    p)
            set_cache(sel_model, "metrics",      m)
            set_cache(sel_model, "dates",        d)
            set_cache(sel_model, "train_actual", ta)
            set_cache(sel_model, "train_index",  ti)
            # Invalidate any cached Gemini summary for this model
            st.session_state.groq_result.pop(sel_model, None)
            st.success(f"✓ {MODEL_LABELS[sel_model]} trained — {len(a)} samples")

    if st.button("▶  Run Forecast", use_container_width=True):
        with st.spinner(f"Forecasting {steps} steps…"):
            res, err = api_get(f"/forecast/{sel_model}", params={"steps": steps}, timeout=120)
        if err:
            st.error(err)
        else:
            fc, fd = parse_forecast(res)
            if fc:
                set_cache(sel_model, "forecast",       fc)
                set_cache(sel_model, "forecast_steps", steps)
                set_cache(sel_model, "forecast_dates", fd)
                st.session_state.groq_result.pop(sel_model, None)
                st.success(f"✓ {len(fc)}-step forecast ready!")
            else:
                st.warning("Forecast returned empty. Check API response keys.")

    st.markdown(f"<hr style='border-color:{BORDER}'>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-family:monospace;font-size:9px;color:{TEXT_DIM};"
        f"text-align:center'>FastAPI · Plotly · Streamlit</p>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# PAGE HEADER
# ════════════════════════════════════════════════════════════════
hdr1, hdr2 = st.columns([5, 1])
with hdr1:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:4px">
       <div style="background:{ACCENT_BLUE};border-radius:12px;width:44px;height:44px;
                   display:flex;align-items:center;justify-content:center;font-size:22px">📊</div>
       <div>
        <h1 style="margin:0;font-size:22px;color:{TEXT}">Time Series Forecasting Dashboard</h1>
        <p style="margin:0;color:{TEXT_MUTED};font-size:13px">
          UHV Plasma Research · Impurity Level Monitor · Analyse and compare forecasting models
        </p>
      </div>
    </div>""", unsafe_allow_html=True)
with hdr2:
    st.markdown(
        f"<div style='text-align:right;padding-top:12px'>"
        f"<span style='background:{mc}18;color:{mc};border:1px solid {mc}30;"
        f"border-radius:20px;padding:4px 12px;font-family:monospace;font-size:11px'>"
        f"{MODEL_LABELS[sel_model]}</span></div>", unsafe_allow_html=True)

st.markdown(f"<hr style='border-color:{BORDER};margin:12px 0 20px'>", unsafe_allow_html=True)

# ── Active data from cache ───────────────────────────────────────
cache     = get_cache(sel_model)
actual    = cache.get("actual",    [])
predicted = cache.get("predicted", [])
metrics   = cache.get("metrics",   {})
forecast  = cache.get("forecast",  [])
f_steps   = cache.get("forecast_steps", steps)
dates     = cache.get("dates", [])
f_dates   = cache.get("forecast_dates", [])
train_act = cache.get("train_actual", [])
train_idx = cache.get("train_index",  [])

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Forecast", "Residuals",
    "Compare Models", "🔍  Insights",
])

# ── TAB 1  Overview ─────────────────────────────────────────────
with tab1:
    mae_v  = metrics.get("mae")
    rmse_v = metrics.get("rmse")
    mape_v = metrics.get("mape")
    acc_v  = metrics.get("accuracy")

    # ── 4-column metric row ──────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(mcard("Mean Absolute Error",
                           f"{mae_v:.4f}"  if mae_v  is not None else "—",
                           "Avg absolute difference", ACCENT_ORANGE, "↗"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown(mcard("Root Mean Sq. Error",
                           f"{rmse_v:.4f}" if rmse_v is not None else "—",
                           "Std dev of errors", ACCENT_PINK, "↗"),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(mcard("Mean Abs % Error",
                           f"{mape_v:.2f}%" if mape_v is not None else "—",
                           "Percentage prediction error", ACCENT_BLUE, "◎"),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(mcard("Forecast Accuracy",
                           f"{acc_v:.2f}%"  if acc_v  is not None else "—",
                           "100% − MAPE  ·  higher is better", ACCENT_GREEN, "✓"),
                    unsafe_allow_html=True)

    st.markdown("###")
    if actual and predicted:
        st.plotly_chart(
            chart_actpred(actual, predicted, sel_model,
                          dates=dates, train_act=train_act, train_idx=train_idx),
            use_container_width=True, key="tab1_actpred")
    else:
        st.info(f"Click **⚙ Train Model** in the sidebar to load "
                f"{MODEL_LABELS[sel_model]} results.")

    if actual:
        vals = pd.Series(actual)
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1: st.metric("Min",    f"{vals.min():.2f}")
        with s2: st.metric("Max",    f"{vals.max():.2f}")
        with s3: st.metric("Mean",   f"{vals.mean():.2f}")
        with s4: st.metric("Median", f"{vals.median():.2f}")
        with s5: st.metric("Std",    f"{vals.std():.2f}")

# ── TAB 2  Forecast ─────────────────────────────────────────────
with tab2:
    cl, cr = st.columns(2)
    with cl:
        st.markdown(
            f"<h3 style='color:{TEXT};font-size:15px'>{MODEL_LABELS[sel_model]} Forecast</h3>"
            f"<p style='color:{TEXT_MUTED};font-size:12px;margin-top:-10px'>"
            f"Actual vs Predicted Values Over Time</p>",
            unsafe_allow_html=True)
        if actual and predicted:
            st.plotly_chart(
                chart_actpred(actual, predicted, sel_model,
                              dates=dates, train_act=train_act, train_idx=train_idx),
                use_container_width=True, key="tab2_actpred")
        else:
            st.info("Train the model first using the sidebar button.")

    with cr:
        st.markdown(
            f"<h3 style='color:{TEXT};font-size:15px'>Future Forecast — {f_steps} Steps</h3>"
            f"<p style='color:{TEXT_MUTED};font-size:12px;margin-top:-10px'>"
            f"{MODEL_LABELS[sel_model]} forward projection</p>",
            unsafe_allow_html=True)
        if forecast:
            st.plotly_chart(
                chart_future(forecast, sel_model, f_steps, dates=f_dates),
                use_container_width=True, key="tab2_future")
        else:
            st.info("Click **▶ Run Forecast** in the sidebar.")

    if forecast:
        st.markdown(f"<h4 style='color:{TEXT}'>Forecast Values Table</h4>",
                    unsafe_allow_html=True)
        df_f = pd.DataFrame({
            "Step": range(1, len(forecast) + 1),
            f"{MODEL_LABELS[sel_model]} Forecast": [round(v, 4) for v in forecast],
        })
        st.dataframe(df_f, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download Forecast CSV", df_f.to_csv(index=False),
                           file_name=f"{sel_model}_forecast_{f_steps}steps.csv",
                           mime="text/csv", use_container_width=True)

# ── TAB 3  Residuals ────────────────────────────────────────────
with tab3:
    if actual and predicted:
        st.plotly_chart(
            chart_residuals(actual, predicted, sel_model, dates=dates),
            use_container_width=True, key="tab3_residuals")
        n   = min(len(actual), len(predicted))
        res = [actual[i] - predicted[i] for i in range(n)]
        r1, r2, r3, r4 = st.columns(4)
        with r1: st.markdown(mcard("Mean Error",       f"{sum(res)/len(res):.4f}", "Average bias",         ACCENT_BLUE,   "◈"), unsafe_allow_html=True)
        with r2: st.markdown(mcard("Max Overestimate", f"{max(res):.4f}",          "Largest +ve residual", ACCENT_ORANGE, "▲"), unsafe_allow_html=True)
        with r3: st.markdown(mcard("Max Underestimate",f"{min(res):.4f}",          "Largest -ve residual", ACCENT_PINK,   "▼"), unsafe_allow_html=True)
        with r4: st.markdown(mcard("Std of Residuals", f"{pd.Series(res).std():.4f}", "Spread of errors",  ACCENT_GREEN,  "◉"), unsafe_allow_html=True)
    else:
        st.info("Train a model to see residual analysis.")

# ── TAB 4  Compare Models ───────────────────────────────────────
with tab4:
    st.markdown(
        f"<h3 style='color:{TEXT};font-size:16px'>⚖ Model Comparison</h3>"
        f"<p style='color:{TEXT_MUTED};font-size:12px;margin-top:-8px'>"
        f"Compare performance metrics between two models</p>",
        unsafe_allow_html=True)

    dc1, dc2, dc3 = st.columns([2, 2, 1])
    with dc1:
        st.markdown(f"<p style='font-family:monospace;font-size:10px;color:{TEXT_MUTED};"
                    f"letter-spacing:1px;text-transform:uppercase'>Model 1</p>",
                    unsafe_allow_html=True)
        cmp1 = st.selectbox("", MODELS, index=0, key="cmp1",
                             label_visibility="collapsed",
                             format_func=lambda x: MODEL_LABELS[x])
    with dc2:
        st.markdown(f"<p style='font-family:monospace;font-size:10px;color:{TEXT_MUTED};"
                    f"letter-spacing:1px;text-transform:uppercase'>Model 2</p>",
                    unsafe_allow_html=True)
        remaining = [m for m in MODELS if m != cmp1]
        cmp2 = st.selectbox("", remaining, index=0, key="cmp2",
                             label_visibility="collapsed",
                             format_func=lambda x: MODEL_LABELS[x])
    with dc3:
        st.markdown("<p style='margin-bottom:8px'>&nbsp;</p>", unsafe_allow_html=True)
        run_cmp = st.button("▶ Compare", use_container_width=True, key="run_cmp")

    if run_cmp:
        if cmp1 == cmp2:
            st.warning("Please select two different models.")
        else:
            prog      = st.progress(0, text="Starting…")
            has_error = False
            for i, m in enumerate([cmp1, cmp2]):
                prog.progress(i * 50, text=f"Training {MODEL_LABELS[m]}…")
                res, err = api_get(f"/train/{m}", timeout=600)
                if err:
                    st.error(f"{MODEL_LABELS[m]}: {err}")
                    has_error = True
                else:
                    a, p, met, d, ta, ti = parse_train(res)
                    set_cache(m, "actual",       a)
                    set_cache(m, "predicted",    p)
                    set_cache(m, "metrics",      met)
                    set_cache(m, "dates",        d)
                    set_cache(m, "train_actual", ta)
                    set_cache(m, "train_index",  ti)
            prog.progress(100, text="Done!"); time.sleep(0.4); prog.empty()
            if not has_error:
                st.success(f"✓ {MODEL_LABELS[cmp1]} and {MODEL_LABELS[cmp2]} compared!")

    d1, d2     = get_cache(cmp1), get_cache(cmp2)
    met1, met2 = d1.get("metrics", {}), d2.get("metrics", {})

    if met1 or met2:
        r1v  = met1.get("rmse", float("inf"))
        r2v  = met2.get("rmse", float("inf"))
        best = cmp1 if r1v <= r2v else cmp2
        bc   = MODEL_COLORS.get(best, ACCENT_GREEN)
        st.markdown(
            f"<div style='background:{BG_CARD};border:1px solid {BORDER};border-left:3px solid {bc};"
            f"border-radius:10px;padding:14px 20px;margin:16px 0'>"
            f"<span style='font-family:monospace;font-size:10px;color:{TEXT_MUTED};"
            f"letter-spacing:1px;text-transform:uppercase'>Best Model</span><br>"
            f"<span style='font-size:22px;font-weight:700;color:{bc}'>{MODEL_LABELS[best]}</span>"
            f"&nbsp;<span style='background:{bc}18;color:{bc};border:1px solid {bc}40;"
            f"border-radius:20px;padding:3px 10px;font-family:monospace;font-size:11px'>"
            f"◆ Lowest RMSE</span></div>",
            unsafe_allow_html=True)

        if met1 and met2:
            st.plotly_chart(chart_compare_bar(cmp1, met1, cmp2, met2),
                            use_container_width=True, key="tab4_compare_bar")

        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown(summary_card(cmp1, met1, MODEL_COLORS.get(cmp1, ACCENT_BLUE), best == cmp1),
                        unsafe_allow_html=True)
        with sc2:
            st.markdown(summary_card(cmp2, met2, MODEL_COLORS.get(cmp2, ACCENT_GREEN), best == cmp2),
                        unsafe_allow_html=True)

        if d1.get("predicted") and d2.get("predicted"):
            st.markdown(f"<h4 style='color:{TEXT};margin-top:24px'>Forecast Overlay</h4>",
                        unsafe_allow_html=True)
            st.plotly_chart(chart_overlay(cmp1, d1, cmp2, d2),
                            use_container_width=True, key="tab4_overlay")
    else:
        st.info("Select two models and click **▶ Compare** to run side-by-side comparison.")

# ── TAB 5  Insights ─────────────────────────────────────────────
with tab5:
    if not actual:
        st.info("Train a model to see dataset insights.")
    else:
        vals = pd.Series(actual)

        # ── Statistical summary ──────────────────────────────────
        st.markdown(
            f"<h4 style='color:{TEXT}'>Statistical Summary — {MODEL_LABELS[sel_model]}</h4>",
            unsafe_allow_html=True)
        g1, g2, g3, g4, g5 = st.columns(5)
        with g1: st.markdown(mcard("Min",     f"{vals.min():.2f}",    "Minimum value",  ACCENT_BLUE,   "▼"), unsafe_allow_html=True)
        with g2: st.markdown(mcard("Max",     f"{vals.max():.2f}",    "Maximum value",  ACCENT_GREEN,  "▲"), unsafe_allow_html=True)
        with g3: st.markdown(mcard("Mean",    f"{vals.mean():.2f}",   "Average value",  ACCENT_ORANGE, "◈"), unsafe_allow_html=True)
        with g4: st.markdown(mcard("Median",  f"{vals.median():.2f}", "Middle value",   ACCENT_PINK,   "◉"), unsafe_allow_html=True)
        with g5: st.markdown(mcard("Std Dev", f"{vals.std():.2f}",    "Spread of data", TEXT_MUTED,    "◎"), unsafe_allow_html=True)

        st.markdown("###")
        ih1, ih2 = st.columns(2)
        with ih1:
            fig_h = go.Figure(go.Histogram(x=vals, nbinsx=40,
                marker_color=ACCENT_BLUE, opacity=0.75, name="Distribution"))
            fig_h.update_layout(title=dict(text="Value Distribution",
                font=dict(color=TEXT, size=13)), **plot_layout(height=280))
            st.plotly_chart(fig_h, use_container_width=True, key="tab5_histogram")
        with ih2:
            _h = ACCENT_ORANGE.lstrip("#")
            _r, _g, _b = (int(_h[i:i+2], 16) for i in (0, 2, 4))
            rgba_orange = f"rgba({_r},{_g},{_b},0.2)"
            fig_b = go.Figure(go.Box(y=vals, name=MODEL_LABELS[sel_model],
                marker_color=ACCENT_ORANGE, line_color=ACCENT_ORANGE,
                fillcolor=rgba_orange))
            fig_b.update_layout(title=dict(text="Box Plot",
                font=dict(color=TEXT, size=13)), **plot_layout(height=280))
            st.plotly_chart(fig_b, use_container_width=True, key="tab5_boxplot")

        # ── Trend banner ─────────────────────────────────────────
        trend_pct = ((vals.iloc[-1] - vals.iloc[0]) / vals.iloc[0] * 100
                     if vals.iloc[0] != 0 else 0)
        if   trend_pct >  2: trend_lbl, tc = "↗ Uptrend",   ACCENT_GREEN
        elif trend_pct < -2: trend_lbl, tc = "↘ Downtrend", ACCENT_PINK
        else:                trend_lbl, tc = "→ Stable",    ACCENT_YELLOW

        st.markdown(
            f"<div style='background:{BG_CARD};border:1px solid {BORDER};border-left:3px solid {tc};"
            f"border-radius:10px;padding:14px 18px;display:flex;align-items:center;gap:14px'>"
            f"<span style='font-size:26px;color:{tc}'>{trend_lbl.split()[0]}</span>"
            f"<div><p style='margin:0;color:{TEXT};font-family:monospace;font-size:14px;font-weight:600'>"
            f"{trend_lbl}</p>"
            f"<p style='margin:0;color:{TEXT_MUTED};font-family:monospace;font-size:11px'>"
            f"Start: {vals.iloc[0]:.2f} → End: {vals.iloc[-1]:.2f} "
            f"&nbsp;|&nbsp; Change: {trend_pct:+.2f}%</p></div></div>",
            unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════
        # GEMINI AI DIAGNOSTIC SUMMARY
        # ════════════════════════════════════════════════════════
        st.markdown("###")
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px'>"
            f"<span style='font-size:20px'>🤖</span>"
            f"<h4 style='color:{TEXT};margin:0'>AI  Summary</h4>"
            f"<span style='background:{ACCENT_PURPLE}25;color:{ACCENT_PURPLE};"
            f"border:1px solid {ACCENT_PURPLE}40;border-radius:20px;"
            f"padding:2px 10px;font-family:monospace;font-size:10px;letter-spacing:1px'>"
            f"GROQ · LLAMA-3.3</span></div>"
            f"<p style='color:{TEXT_MUTED};font-family:monospace;font-size:11px;margin-bottom:14px'>"
            f"Generate a professional UHV system diagnostic report using the current model results.</p>",
            unsafe_allow_html=True)

        # API key input
        key_col, btn_col = st.columns([4, 1])
        with key_col:
            api_key_input = st.text_input(
                "Groq API Key",
                value=st.session_state.groq_key,
                type="password",
                placeholder="Paste your Groq API key (gsk_…)",
                label_visibility="collapsed",
                key="groq_key_field",
            )
        with btn_col:
            gen_btn = st.button("✨ Analyse", use_container_width=True, key="groq_gen")

        if api_key_input:
            st.session_state.groq_key = api_key_input

        # ── Generate ─────────────────────────────────────────────
        if gen_btn:
            if not st.session_state.groq_key.strip():
                st.warning("Please enter your Groq API key first.")
            elif not actual:
                st.warning("Train a model first so there is data to analyse.")
            else:
                try:
                    with st.spinner("🤖 Groq is analysing your UHV data…"):
                        summary = call_groq(
                            api_key     = st.session_state.groq_key,
                            model_name  = sel_model,
                            metrics_dict= metrics,
                            actual      = actual,
                            forecast_vals= forecast,
                            trend_label = trend_lbl,
                            trend_pct   = trend_pct,
                        )
                    st.session_state.groq_result[sel_model] = summary
                except RuntimeError as e:
                    err = str(e)
                    if "403" in err or "invalid" in err.lower() or "permission" in err.lower():
                        st.error("❌ Invalid GROQ API key. Please check and try again.")
                    elif "429" in err or "quota" in err.lower():
                        st.error("❌ GROQ quota exceeded. Try again later.")
                    else:
                        st.error(f"❌ {err}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")

        # ── Display cached summary ────────────────────────────────
        cached_summary = st.session_state.groq_result.get(sel_model)
        if cached_summary:
            st.markdown(
                f"<div style='background:{BG_CARD};border:1px solid {BORDER};"
                f"border-left:3px solid {ACCENT_PURPLE};border-radius:12px;"
                f"padding:20px 24px;margin-top:12px'>"
                f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:10px'>"
                f"<span style='font-size:16px'>📋</span>"
                f"<span style='font-family:monospace;font-size:11px;color:{TEXT_MUTED};"
                f"letter-spacing:1px;text-transform:uppercase'>"
                f"Diagnostic Report · {MODEL_LABELS[sel_model]}</span></div></div>",
                unsafe_allow_html=True)
            st.markdown(cached_summary)

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇ Download Report (.txt)",
                    data=cached_summary,
                    file_name=f"{sel_model}_diagnostic_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with dl2:
                # Build a quick markdown version for download
                md_report = (
                    f"# UHV Diagnostic Report — {MODEL_LABELS[sel_model]}\n\n"
                    f"**Generated by:** Groq · Llama-3.3-70b\n\n"
                    f"**Accuracy:** {metrics.get('accuracy', 'N/A'):.2f}% | "
                    f"**RMSE:** {metrics.get('rmse', 'N/A'):.4f} | "
                    f"**MAE:** {metrics.get('mae', 'N/A'):.4f}\n\n"
                    f"---\n\n{cached_summary}"
                    if isinstance(metrics.get("accuracy"), float)
                    else f"# UHV Diagnostic Report\n\n{cached_summary}"
                )
                st.download_button(
                    "⬇ Download Report (.md)",
                    data=md_report,
                    file_name=f"{sel_model}_diagnostic_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        # Help text
        st.markdown(
            f"<p style='font-family:monospace;font-size:10px;color:{TEXT_DIM};margin-top:10px'>"
            f"ℹ Get a free key at "
            f"<a href='https://console.groq.com' target='_blank' "
            f"style='color:{ACCENT_BLUE}'>console.groq.com</a> · "
            f"No extra packages needed — uses REST API directly.</p>",
            unsafe_allow_html=True)

        # ── Export raw data ──────────────────────────────────────
        st.markdown("###")
        df_exp = pd.DataFrame({
            "actual":    actual,
            "predicted": predicted[:len(actual)] if predicted else [None] * len(actual),
        })
        st.download_button(
            "⬇ Download Training Data CSV",
            df_exp.to_csv(index=False),
            file_name=f"{sel_model}_training_data.csv",
            mime="text/csv",
        )