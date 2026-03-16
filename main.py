"""
DCF Valuation Tool — Streamlit App
===================================
Run with:  streamlit run main.py

Requirements:
    pip install streamlit yfinance pandas numpy plotly
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DCF Valuation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
}

.main { background: #0d0f14; }

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0d0f14 0%, #131720 50%, #0d1117 100%);
    border: 1px solid #1e2530;
    border-radius: 12px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,212,170,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #e8eaf0;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    color: #4a5568;
    font-size: 0.85rem;
    margin-top: 0.5rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #131720;
    border: 1px solid #1e2530;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.metric-label {
    color: #4a5568;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}
.metric-value {
    color: #e8eaf0;
    font-size: 1.6rem;
    font-weight: 500;
}
.metric-value.green { color: #00d4aa; }
.metric-value.red   { color: #ff6b6b; }
.metric-value.amber { color: #f6ad55; }

/* Verdict banner */
.verdict-undervalued {
    background: linear-gradient(90deg, rgba(0,212,170,0.08), transparent);
    border-left: 3px solid #00d4aa;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.5rem;
    color: #00d4aa;
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    margin: 1rem 0;
}
.verdict-overvalued {
    background: linear-gradient(90deg, rgba(255,107,107,0.08), transparent);
    border-left: 3px solid #ff6b6b;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.5rem;
    color: #ff6b6b;
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    margin: 1rem 0;
}
.verdict-fair {
    background: linear-gradient(90deg, rgba(246,173,85,0.08), transparent);
    border-left: 3px solid #f6ad55;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.5rem;
    color: #f6ad55;
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    margin: 1rem 0;
}

/* Section divider */
.section-title {
    font-family: 'DM Serif Display', serif;
    color: #e8eaf0;
    font-size: 1.4rem;
    border-bottom: 1px solid #1e2530;
    padding-bottom: 0.5rem;
    margin: 2rem 0 1.2rem 0;
}

/* Disclaimer */
.disclaimer {
    background: #0d0f14;
    border: 1px solid #1e2530;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    color: #4a5568;
    font-size: 0.75rem;
    margin-top: 2rem;
    line-height: 1.6;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #0d0f14;
    border-right: 1px solid #1e2530;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CORE LOGIC
# ─────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_company_data(ticker: str) -> dict:
    tk = yf.Ticker(ticker)
    info     = tk.info
    cashflow = tk.cashflow
    balance  = tk.balance_sheet

    if not info or not info.get("regularMarketPrice"):
        raise ValueError(f"No data found for '{ticker}'. Check the symbol.")

    # FCF history
    fcf_history, fcf_years = [], []
    for row_op, row_cap in [
        ("Operating Cash Flow", "Capital Expenditure"),
        ("Total Cash From Operating Activities", "Capital Expenditures"),
    ]:
        try:
            op   = cashflow.loc[row_op]
            cap  = cashflow.loc[row_cap]
            fcf_raw = (op + cap).dropna()
            fcf_history = fcf_raw.values[::-1].tolist()
            fcf_years   = [str(d.year) for d in fcf_raw.index[::-1]]
            break
        except KeyError:
            continue

    # Net debt
    try:    total_debt = float(balance.loc["Total Debt"].iloc[0])
    except: total_debt = float(info.get("totalDebt", 0) or 0)
    try:    cash = float(balance.loc["Cash And Cash Equivalents"].iloc[0])
    except: cash = float(info.get("totalCash", 0) or 0)

    shares = (info.get("sharesOutstanding")
              or info.get("impliedSharesOutstanding")
              or info.get("floatShares") or 1)

    return {
        "ticker":         ticker.upper(),
        "name":           info.get("longName", ticker.upper()),
        "sector":         info.get("sector", "N/A"),
        "industry":       info.get("industry", "N/A"),
        "currency":       info.get("currency", "USD"),
        "current_price":  info.get("regularMarketPrice") or info.get("currentPrice"),
        "market_cap":     info.get("marketCap"),
        "shares":         shares,
        "net_debt":       total_debt - cash,
        "fcf_history":    fcf_history,
        "fcf_years":      fcf_years,
        "pe_ratio":       info.get("trailingPE"),
        "forward_pe":     info.get("forwardPE"),
        "ev_ebitda":      info.get("enterpriseToEbitda"),
        "beta":           info.get("beta"),
        "analyst_target": info.get("targetMeanPrice"),
        "description":    info.get("longBusinessSummary", ""),
    }


def estimate_growth_rate(fcf_history: list, years: int = 5) -> float:
    data = [f for f in fcf_history if f and not np.isnan(f)]
    if len(data) < 2:
        return 0.08
    data = data[-(years + 1):]
    start, end, n = data[0], data[-1], len(data) - 1
    if start <= 0 or end <= 0:
        pos = [f for f in data if f > 0]
        return ((pos[-1] / pos[0]) ** (1 / (len(pos) - 1)) - 1) if len(pos) >= 2 else 0.08
    return max(min((end / start) ** (1 / n) - 1, 0.40), -0.15)


def run_dcf(base_fcf, growth_rate, wacc, terminal_growth, proj_years, net_debt, shares):
    years    = list(range(1, proj_years + 1))
    proj_fcf = [base_fcf * (1 + growth_rate) ** y for y in years]
    disc_fcf = [fcf / (1 + wacc) ** y for y, fcf in zip(years, proj_fcf)]

    terminal_fcf  = proj_fcf[-1] * (1 + terminal_growth)
    terminal_val  = terminal_fcf / (wacc - terminal_growth)
    disc_terminal = terminal_val / (1 + wacc) ** proj_years

    pv_fcf       = sum(disc_fcf)
    enterprise_v = pv_fcf + disc_terminal
    equity_v     = enterprise_v - net_debt
    intrinsic    = equity_v / shares if shares else 0

    return {
        "years":               years,
        "proj_fcf":            proj_fcf,
        "disc_fcf":            disc_fcf,
        "pv_fcf":              pv_fcf,
        "terminal_val":        terminal_val,
        "disc_terminal":       disc_terminal,
        "enterprise_value":    enterprise_v,
        "equity_value":        equity_v,
        "intrinsic_per_share": intrinsic,
    }


def sensitivity_matrix(base_fcf, growth_rate, net_debt, shares, proj_years,
                        wacc_range, tgr_range):
    matrix = []
    for tgr in tgr_range:
        row = []
        for wacc in wacc_range:
            if wacc <= tgr:
                row.append(np.nan)
            else:
                r = run_dcf(base_fcf, growth_rate, wacc, tgr, proj_years, net_debt, shares)
                row.append(r["intrinsic_per_share"])
        matrix.append(row)
    return np.array(matrix)


# ─────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#8892a4"),
    xaxis=dict(gridcolor="#1e2530", zerolinecolor="#1e2530"),
    yaxis=dict(gridcolor="#1e2530", zerolinecolor="#1e2530"),
    margin=dict(l=10, r=10, t=40, b=10),
)


def chart_fcf_history(fcf_history, fcf_years):
    colors = ["#00d4aa" if f > 0 else "#ff6b6b" for f in fcf_history]
    fig = go.Figure(go.Bar(
        x=fcf_years,
        y=[f / 1e9 for f in fcf_history],
        marker_color=colors,
        text=[f"${f/1e9:.2f}B" for f in fcf_history],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(**CHART_THEME, title="Historical Free Cash Flow (USD Billions)",
                      title_font=dict(size=14, color="#e8eaf0"))
    return fig


def chart_projected_fcf(result):
    years = [f"Year {y}" for y in result["years"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Projected FCF",
        x=years, y=[f / 1e9 for f in result["proj_fcf"]],
        marker_color="#2563eb", opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        name="Discounted FCF",
        x=years, y=[f / 1e9 for f in result["disc_fcf"]],
        marker_color="#00d4aa",
    ))
    fig.update_layout(**CHART_THEME, barmode="group",
                      title="Projected vs Discounted FCF (USD Billions)",
                      title_font=dict(size=14, color="#e8eaf0"),
                      legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2530"))
    return fig


def chart_value_waterfall(result, net_debt):
    labels = ["PV of FCFs", "Terminal Value (PV)", "Enterprise Value", "Less: Net Debt", "Equity Value"]
    values = [
        result["pv_fcf"] / 1e9,
        result["disc_terminal"] / 1e9,
        result["enterprise_value"] / 1e9,
        -net_debt / 1e9,
        result["equity_value"] / 1e9,
    ]
    colors = ["#2563eb", "#00d4aa", "#e8eaf0", "#ff6b6b", "#00d4aa"]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"${v:.2f}B" for v in values],
        textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(**CHART_THEME, title="Value Bridge (USD Billions)",
                      title_font=dict(size=14, color="#e8eaf0"))
    return fig


def chart_sensitivity(matrix, wacc_range, tgr_range, current_price):
    wacc_labels = [f"{w*100:.1f}%" for w in wacc_range]
    tgr_labels  = [f"{t*100:.1f}%" for t in tgr_range]

    # Colour relative to current price
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=wacc_labels,
        y=tgr_labels,
        colorscale=[
            [0.0,  "#ff6b6b"],
            [0.5,  "#f6ad55"],
            [1.0,  "#00d4aa"],
        ],
        zmid=current_price,
        text=[[f"${v:.2f}" if not np.isnan(v) else "N/A" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="WACC: %{x}<br>TGR: %{y}<br>Intrinsic: %{text}<extra></extra>",
        colorbar=dict(title="$/share", tickfont=dict(color="#8892a4")),
    ))
    fig.update_layout(
        **CHART_THEME,
        title="Sensitivity: Intrinsic Value per Share (WACC × Terminal Growth Rate)",
        title_font=dict(size=14, color="#e8eaf0"),
        xaxis_title="WACC →",
        yaxis_title="Terminal Growth Rate →",
    )
    return fig


def chart_price_vs_intrinsic(current_price, intrinsic, mos_price, analyst_target):
    labels, values, colors = [], [], []
    entries = [
        ("MoS Price", mos_price, "#4a5568"),
        ("Current Price", current_price, "#2563eb"),
        ("Intrinsic Value", intrinsic, "#00d4aa"),
    ]
    if analyst_target:
        entries.append(("Analyst Target", analyst_target, "#f6ad55"))
    entries.sort(key=lambda x: x[1])
    for l, v, c in entries:
        labels.append(l); values.append(v); colors.append(c)

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"${v:,.2f}" for v in values],
        textposition="outside",
        textfont=dict(size=12, color="#e8eaf0"),
    ))
    fig.update_layout(**CHART_THEME, title="Price Comparison",
                      title_font=dict(size=14, color="#e8eaf0"),
                      yaxis_title="USD per Share")
    return fig


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def fmt_b(v):
    if v is None or np.isnan(v): return "N/A"
    return f"${v/1e9:,.2f}B"

def fmt_m(v):
    if v is None or np.isnan(v): return "N/A"
    return f"${v/1e6:,.2f}M"

def fmt_p(v):
    if v is None or np.isnan(v): return "N/A"
    return f"${v:,.2f}"

def fmt_pct(v):
    if v is None or np.isnan(v): return "N/A"
    return f"{v*100:.2f}%"


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 0.5rem;'>
        <div style='font-family: DM Serif Display, serif; font-size: 1.4rem; color: #e8eaf0;'>⟁ DCF Tool</div>
        <div style='color: #4a5568; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;'>Discounted Cash Flow</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    ticker_input = st.text_input(
        "Ticker Symbol",
        value="AAPL",
        placeholder="e.g. AAPL, MSFT, SHEL.L",
        help="Any Yahoo Finance ticker. UK stocks: add .L (e.g. BP.L)"
    ).strip().upper()

    run_btn = st.button("▶  Run Valuation", type="primary", use_container_width=True)

    st.divider()
    st.markdown("<div style='color:#4a5568;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;'>DCF Assumptions</div>", unsafe_allow_html=True)

    wacc = st.slider("WACC (%)", 5.0, 20.0, 10.0, 0.5) / 100
    terminal_growth = st.slider("Terminal Growth Rate (%)", 0.5, 5.0, 2.5, 0.25) / 100
    proj_years = st.slider("Projection Years", 3, 10, 5, 1)
    mos = st.slider("Margin of Safety (%)", 0, 40, 20, 5) / 100

    st.divider()
    st.markdown("<div style='color:#4a5568;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;'>Override Growth Rate</div>", unsafe_allow_html=True)
    override_growth = st.checkbox("Manually set FCF growth rate")
    manual_growth = None
    if override_growth:
        manual_growth = st.slider("FCF Growth Rate (%)", -10.0, 40.0, 10.0, 0.5) / 100

    st.divider()
    st.markdown("<div style='color:#4a5568; font-size:0.72rem;'>Data via Yahoo Finance · Not financial advice</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-title'>DCF Valuation</div>
    <div class='hero-sub'>Discounted Cash Flow · Intrinsic Value · Sensitivity Analysis</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN OUTPUT
# ─────────────────────────────────────────────
if run_btn or "dcf_data" in st.session_state:

    if run_btn:
        with st.spinner(f"Fetching data for {ticker_input} …"):
            try:
                data = fetch_company_data(ticker_input)
                st.session_state["dcf_data"] = data
                st.session_state["dcf_ticker"] = ticker_input
            except Exception as e:
                st.error(f"❌ {e}")
                st.stop()
    else:
        data = st.session_state["dcf_data"]

    # ── Validate FCF ────────────────────────────────────────────────────────
    if not data["fcf_history"]:
        st.error("Could not retrieve FCF data for this ticker. Try a different company.")
        st.stop()

    base_fcf_auto = data["fcf_history"][-1]
    growth_auto   = estimate_growth_rate(data["fcf_history"])
    growth_rate   = manual_growth if manual_growth is not None else growth_auto

    if base_fcf_auto <= 0:
        st.warning(f"⚠ Most recent FCF is negative ({fmt_m(base_fcf_auto)}). "
                   "DCF requires positive FCF — using average of positive historical values.")
        pos = [f for f in data["fcf_history"] if f > 0]
        if not pos:
            st.error("All historical FCF values are negative. Cannot run a meaningful DCF.")
            st.stop()
        base_fcf = np.mean(pos)
    else:
        base_fcf = base_fcf_auto

    if wacc <= terminal_growth:
        st.error("WACC must be greater than the Terminal Growth Rate.")
        st.stop()

    # ── Run DCF ─────────────────────────────────────────────────────────────
    result = run_dcf(base_fcf, growth_rate, wacc, terminal_growth, proj_years,
                     data["net_debt"], data["shares"])
    intrinsic  = result["intrinsic_per_share"]
    mos_price  = intrinsic * (1 - mos)
    price      = data["current_price"]
    upside     = (intrinsic - price) / price

    # ── Company Header ───────────────────────────────────────────────────────
    st.markdown(f"<div class='section-title'>{data['name']} &nbsp;·&nbsp; <span style='color:#4a5568;font-size:0.9rem;'>{data['sector']} · {data['industry']}</span></div>", unsafe_allow_html=True)

    if data["description"]:
        with st.expander("Company Description"):
            st.write(data["description"][:600] + "…" if len(data["description"]) > 600 else data["description"])

    # ── Verdict ─────────────────────────────────────────────────────────────
    if upside > 0.25:
        verdict_cls = "verdict-undervalued"
        verdict_txt = f"✔  Potentially Undervalued — intrinsic value is {upside*100:.1f}% above the current price"
    elif upside < -0.25:
        verdict_cls = "verdict-overvalued"
        verdict_txt = f"✘  Potentially Overvalued — intrinsic value is {abs(upside)*100:.1f}% below the current price"
    else:
        verdict_cls = "verdict-fair"
        verdict_txt = f"~  Fairly Valued — intrinsic value is within 25% of the current price ({upside*100:+.1f}%)"

    st.markdown(f"<div class='{verdict_cls}'>{verdict_txt}</div>", unsafe_allow_html=True)

    # ── Key Metrics Row ─────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    upside_color = "green" if upside > 0 else "red"

    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Current Price</div>
            <div class='metric-value'>{fmt_p(price)}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Intrinsic Value</div>
            <div class='metric-value green'>{fmt_p(intrinsic)}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>MoS Price ({mos*100:.0f}%)</div>
            <div class='metric-value'>{fmt_p(mos_price)}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Upside / Downside</div>
            <div class='metric-value {upside_color}'>{upside*100:+.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Market Cap</div>
            <div class='metric-value'>{fmt_b(data['market_cap'])}</div>
        </div>""", unsafe_allow_html=True)

    # ── Comparables Row ─────────────────────────────────────────────────────
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>P/E (trailing)</div>
            <div class='metric-value'>{f"{data['pe_ratio']:.1f}×" if data['pe_ratio'] else "N/A"}</div>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>EV / EBITDA</div>
            <div class='metric-value'>{f"{data['ev_ebitda']:.1f}×" if data['ev_ebitda'] else "N/A"}</div>
        </div>""", unsafe_allow_html=True)
    with d3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Beta</div>
            <div class='metric-value'>{f"{data['beta']:.2f}" if data['beta'] else "N/A"}</div>
        </div>""", unsafe_allow_html=True)
    with d4:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>Analyst Target</div>
            <div class='metric-value amber'>{fmt_p(data['analyst_target'])}</div>
        </div>""", unsafe_allow_html=True)

    # ── Charts Row 1 ────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Cash Flow Analysis</div>", unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)
    with ch1:
        if data["fcf_history"] and data["fcf_years"]:
            st.plotly_chart(chart_fcf_history(data["fcf_history"], data["fcf_years"]),
                            use_container_width=True)
        else:
            st.info("No historical FCF data available.")
    with ch2:
        st.plotly_chart(chart_projected_fcf(result), use_container_width=True)

    # ── Charts Row 2 ────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Valuation</div>", unsafe_allow_html=True)
    cv1, cv2 = st.columns(2)
    with cv1:
        st.plotly_chart(chart_value_waterfall(result, data["net_debt"]),
                        use_container_width=True)
    with cv2:
        st.plotly_chart(chart_price_vs_intrinsic(price, intrinsic, mos_price, data["analyst_target"]),
                        use_container_width=True)

    # ── Sensitivity Heatmap ─────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Sensitivity Analysis</div>", unsafe_allow_html=True)
    wacc_range = [wacc - 0.02, wacc - 0.01, wacc, wacc + 0.01, wacc + 0.02]
    wacc_range = [max(w, 0.01) for w in wacc_range]
    tgr_range  = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]
    tgr_range  = [t for t in tgr_range if t < wacc]

    mat = sensitivity_matrix(base_fcf, growth_rate, data["net_debt"],
                             data["shares"], proj_years, wacc_range, tgr_range)
    st.plotly_chart(chart_sensitivity(mat, wacc_range, tgr_range, price),
                    use_container_width=True)
    st.caption("🟢 Green = above current price  ·  🔴 Red = below current price  ·  Cells show intrinsic value per share")

    # ── DCF Detail Table ────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>Projected Cash Flow Detail</div>", unsafe_allow_html=True)
    cf_df = pd.DataFrame({
        "Year":             [f"Year {y}" for y in result["years"]],
        "Projected FCF":    [fmt_m(f) for f in result["proj_fcf"]],
        "Discounted FCF":   [fmt_m(f) for f in result["disc_fcf"]],
    })
    st.dataframe(cf_df, use_container_width=True, hide_index=True)

    summary_df = pd.DataFrame({
        "Component": ["PV of FCFs", "Terminal Value (undiscounted)", "Terminal Value (discounted)",
                      "Enterprise Value", "Less: Net Debt", "Equity Value", "Intrinsic Value / Share"],
        "Value": [fmt_m(result["pv_fcf"]), fmt_m(result["terminal_val"]),
                  fmt_m(result["disc_terminal"]), fmt_m(result["enterprise_value"]),
                  fmt_m(data["net_debt"]), fmt_m(result["equity_value"]),
                  fmt_p(intrinsic)],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── Assumptions Summary ─────────────────────────────────────────────────
    with st.expander("📋  Full Assumptions Used"):
        assump_df = pd.DataFrame({
            "Parameter": ["Base FCF", "FCF Growth Rate", "WACC", "Terminal Growth Rate",
                          "Projection Years", "Margin of Safety", "Net Debt", "Shares Outstanding"],
            "Value": [fmt_m(base_fcf), fmt_pct(growth_rate), fmt_pct(wacc),
                      fmt_pct(terminal_growth), str(proj_years), fmt_pct(mos),
                      fmt_m(data["net_debt"]), f"{data['shares']/1e6:,.1f}M"],
        })
        st.dataframe(assump_df, use_container_width=True, hide_index=True)

    # ── Disclaimer ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class='disclaimer'>
        ⚠ <strong>Disclaimer:</strong> This tool is for educational and research purposes only.
        DCF models are highly sensitive to input assumptions and should not be used as the sole basis
        for investment decisions. Always conduct your own due diligence and consult a qualified
        financial advisor before investing. Past performance is not indicative of future results.
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Landing state ───────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 4rem 2rem; color: #4a5568;'>
        <div style='font-size: 3rem; margin-bottom: 1rem;'>📊</div>
        <div style='font-family: DM Serif Display, serif; font-size: 1.6rem; color: #8892a4; margin-bottom: 0.8rem;'>
            Enter a ticker and run your valuation
        </div>
        <div style='font-size: 0.85rem; line-height: 1.8;'>
            Type any ticker in the sidebar (e.g. <code>AAPL</code>, <code>MSFT</code>, <code>NVDA</code>, <code>SHEL.L</code>)<br>
            Adjust the DCF assumptions · Click <strong>Run Valuation</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
