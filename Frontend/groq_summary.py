import json
import requests
import statistics
 
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"   # free on Groq
 
 
def _build_prompt(
    model_name: str,
    actual: list,
    predicted: list,
    metrics: dict,
    forecast: list,
    forecast_steps: int,
) -> str:
    # ── Metrics block ────────────────────────────────────────────
    mae_v  = metrics.get("mae")
    rmse_v = metrics.get("rmse")
    mape_v = metrics.get("mape")
    acc_v  = metrics.get("accuracy")
 
    parts = []
    if mae_v  is not None: parts.append(f"MAE: {mae_v:.4f}")
    if rmse_v is not None: parts.append(f"RMSE: {rmse_v:.4f}")
    if mape_v is not None: parts.append(f"MAPE: {mape_v:.2f}%")
    if acc_v  is not None: parts.append(f"Accuracy: {acc_v:.2f}%")
    metrics_block = ", ".join(parts) if parts else "No metrics available"
 
    # ── Data stats block ─────────────────────────────────────────
    n = len(actual)
    if n > 0:
        mn        = min(actual)
        mx        = max(actual)
        mean      = sum(actual) / n
        med       = statistics.median(actual)
        std       = statistics.stdev(actual) if n > 1 else 0.0
        trend_pct = ((actual[-1] - actual[0]) / actual[0] * 100) if actual[0] != 0 else 0
        data_block = (
            f"Points: {n}, Min: {mn:.2f}, Max: {mx:.2f}, "
            f"Mean: {mean:.2f}, Median: {med:.2f}, Std: {std:.2f}, "
            f"Trend: {trend_pct:+.2f}%"
        )
    else:
        data_block = "No actual data available."
 
    # ── Residuals block ──────────────────────────────────────────
    if actual and predicted:
        k         = min(len(actual), len(predicted))
        res       = [actual[i] - predicted[i] for i in range(k)]
        mean_res  = sum(res) / k
        residuals_block = (
            f"Mean residual: {mean_res:.4f}, "
            f"Max over: {max(res):.4f}, Max under: {min(res):.4f}"
        )
    else:
        residuals_block = "No residual data available."
 
    # ── Forecast block ───────────────────────────────────────────
    if forecast:
        fc_mean  = sum(forecast) / len(forecast)
        fc_trend = ((forecast[-1] - forecast[0]) / forecast[0] * 100) if forecast[0] != 0 else 0
        forecast_block = (
            f"{forecast_steps}-step forecast. "
            f"Mean: {fc_mean:.2f}, Trend: {fc_trend:+.2f}%"
        )
    else:
        forecast_block = "No forecast generated yet."
 
    return f"""You are a senior systems monitoring analyst specializing in time-series sensor data from industrial and scientific environments. \
Your role is to interpret forecasting model outputs in terms of system reliability, operational stability, and predictive maintenance value. \
The dataset contains hourly-aggregated sensor readings ("Count") that reflect system activity, load patterns, or measurable process behavior over time. \
Two models have already been compared side-by-side in the dashboard's Compare Models tab; this report focuses on {model_name.upper()} and its standing relative to the other model.
 
Analyze the following results and provide a concise, professional structured summary.
 
MODEL: {model_name.upper()}
METRICS: {metrics_block}
DATA (hourly sensor readings): {data_block}
RESIDUALS: {residuals_block}
FORECAST: {forecast_block}
 
Respond with exactly these five sections:
 
**🎯 Model Performance**
Assess the forecasting accuracy of {model_name.upper()} in the context of reliable sensor prediction — evaluate whether the MAE and RMSE are within acceptable bounds for operational monitoring, and whether MAPE reflects a model trustworthy enough for automated alerting or intervention.
 
**📈 Data Insights**
Characterize the sensor signal in terms of stability and operational behavior — comment on the overall trend direction, the degree of variability (spread between min/max and std deviation), and whether the system appears to be operating in a stable, degrading, or escalating state.
 
**🔍 Model Comparison & Mathematical Analysis**
Based on the data characteristics above (trend direction, std deviation, signal range, and residual structure), reason about why {model_name.upper()} performs the way it does relative to the other model compared in the dashboard. \
Use the following mathematical intuitions: ARIMA/SARIMA are linear autoregressive models that perform best on stationary or weakly-trended signals with stable autocorrelation — high std or non-linear dynamics will cause them to underfit; \
LSTM captures long-range non-linear temporal dependencies via gating mechanisms (forget/input/output gates) and performs well on complex or irregular signals but needs sufficient data volume and may overfit on short series; \
Prophet decomposes signal into trend + Fourier-series seasonality components and handles missing data well but is sensitive to noise and performs poorly when no strong seasonal pattern exists; \
Random Forest builds an ensemble on lagged features and handles non-linearity but treats predictions independently, losing temporal continuity on smooth or slowly-drifting signals. \
Conclude with which model's mathematical assumptions best match the observed data characteristics and why.
 
**🔮 Forecast Outlook**
Interpret the {forecast_steps}-step forecast in terms of practical utility — describe the predicted trajectory (stable, rising, or declining sensor counts), assess whether the forecast trend is consistent with historical behavior, and indicate how confidently this forecast can guide operational decisions.
 
**💡 Recommendations**
Give 2–3 concrete, actionable suggestions — such as switching to the better-suited model based on the mathematical analysis above, setting dynamic alerting thresholds from the forecast range, or scheduling retraining if residuals show systematic drift.
 
Keep each section to 2–3 sentences. Be direct, analytical, and grounded in sensor monitoring practice. Avoid generic statements."""
 
 
def get_gemini_summary(
    model_name: str,
    actual: list,
    predicted: list,
    metrics: dict,
    forecast: list,
    forecast_steps: int,
    api_key: str,
) -> tuple:
    """
    Calls Groq API (llama-3.3-70b) via OpenAI-compatible REST.
    Returns (summary_text, None) on success.
    Returns (None, error_string) on failure.
    Get free key at: https://console.groq.com
    """
    if not api_key or api_key.strip() in ("YOUR_GROQ_API_KEY_HERE", ""):
        return None, "Groq API key not set. Paste your key (starts with gsk_…)."
 
    prompt  = _build_prompt(model_name, actual, predicted, metrics, forecast, forecast_steps)
 
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert time series analyst for UHV plasma research. Be concise and analytical."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.4,
        "max_tokens": 1024,
    }
 
    try:
        resp = requests.post(
            GROQ_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key.strip()}",
            },
            data=json.dumps(payload),
            timeout=30,
        )
        resp.raise_for_status()
        data   = resp.json()
        text   = data["choices"][0]["message"]["content"].strip()
        if not text:
            return None, "Groq returned an empty response."
        return text, None
 
    except requests.exceptions.Timeout:
        return None, "Groq API timed out (30s). Try again."
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code
        if code == 400:
            return None, "Bad request — check your API key."
        if code == 401:
            return None, "API key invalid. Get one at console.groq.com"
        if code == 429:
            return None, "Groq quota exceeded. Wait a moment and retry."
        return None, f"HTTP {code}: {e.response.text[:200]}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"