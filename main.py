"""
Investment Banking Toolkit — Streamlit App
==========================================
Run with:  streamlit run main.py

Requirements:
    pip install streamlit yfinance pandas numpy plotly
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="IB Toolkit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
h1,h2,h3 { font-family: 'DM Serif Display', serif !important; }
.main { background: #0d0f14; }
.hero {
    background: linear-gradient(135deg,#0d0f14 0%,#131720 50%,#0d1117 100%);
    border:1px solid #1e2530; border-radius:12px;
    padding:2.5rem 3rem; margin-bottom:2rem; position:relative; overflow:hidden;
}
.hero::before {
    content:''; position:absolute; top:-50%; right:-10%;
    width:400px; height:400px;
    background:radial-gradient(circle,rgba(0,212,170,0.06) 0%,transparent 70%);
    pointer-events:none;
}
.hero-title { font-family:'DM Serif Display',serif; font-size:2.8rem; color:#e8eaf0; margin:0; line-height:1.1; }
.hero-sub   { color:#4a5568; font-size:0.85rem; margin-top:0.5rem; letter-spacing:0.12em; text-transform:uppercase; }
.metric-card { background:#131720; border:1px solid #1e2530; border-radius:10px; padding:1.2rem 1.5rem; margin-bottom:1rem; }
.metric-label { color:#4a5568; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.3rem; }
.metric-value { color:#e8eaf0; font-size:1.6rem; font-weight:500; }
.metric-value.green { color:#00d4aa; }
.metric-value.red   { color:#ff6b6b; }
.metric-value.amber { color:#f6ad55; }
.verdict-undervalued { background:linear-gradient(90deg,rgba(0,212,170,0.08),transparent); border-left:3px solid #00d4aa; border-radius:0 8px 8px 0; padding:1rem 1.5rem; color:#00d4aa; font-family:'DM Serif Display',serif; font-size:1.1rem; margin:1rem 0; }
.verdict-overvalued  { background:linear-gradient(90deg,rgba(255,107,107,0.08),transparent); border-left:3px solid #ff6b6b; border-radius:0 8px 8px 0; padding:1rem 1.5rem; color:#ff6b6b; font-family:'DM Serif Display',serif; font-size:1.1rem; margin:1rem 0; }
.verdict-fair        { background:linear-gradient(90deg,rgba(246,173,85,0.08),transparent); border-left:3px solid #f6ad55; border-radius:0 8px 8px 0; padding:1rem 1.5rem; color:#f6ad55; font-family:'DM Serif Display',serif; font-size:1.1rem; margin:1rem 0; }
.section-title { font-family:'DM Serif Display',serif; color:#e8eaf0; font-size:1.4rem; border-bottom:1px solid #1e2530; padding-bottom:0.5rem; margin:2rem 0 1.2rem 0; }
.disclaimer { background:#0d0f14; border:1px solid #1e2530; border-radius:8px; padding:1rem 1.5rem; color:#4a5568; font-size:0.75rem; margin-top:2rem; line-height:1.6; }
section[data-testid="stSidebar"] { background:#0d0f14; border-right:1px solid #1e2530; }
</style>
""", unsafe_allow_html=True)

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#8892a4"),
    xaxis=dict(gridcolor="#1e2530", zerolinecolor="#1e2530"),
    yaxis=dict(gridcolor="#1e2530", zerolinecolor="#1e2530"),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ─────────────────────────────────────────────
#  SHARED HELPERS
# ─────────────────────────────────────────────
def fmt_b(v):
    try:
        if v is None or np.isnan(float(v)): return "N/A"
        return f"${float(v)/1e9:,.2f}B"
    except: return "N/A"

def fmt_m(v):
    try:
        if v is None or np.isnan(float(v)): return "N/A"
        return f"${float(v)/1e6:,.2f}M"
    except: return "N/A"

def fmt_p(v):
    try:
        if v is None or np.isnan(float(v)): return "N/A"
        return f"${float(v):,.2f}"
    except: return "N/A"

def fmt_pct(v):
    try:
        if v is None or np.isnan(float(v)): return "N/A"
        return f"{float(v)*100:.2f}%"
    except: return "N/A"

def safe_float(v):
    try: return float(v)
    except: return np.nan

# ─────────────────────────────────────────────
#  NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem;'>
        <div style='font-family:DM Serif Display,serif;font-size:1.4rem;color:#e8eaf0;'>⟁ IB Toolkit</div>
        <div style='color:#4a5568;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;'>Investment Banking Suite</div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigation", [
        "📊  DCF Valuation",
        "🏢  Comparable Companies",
        "📑  3-Statement Model",
        "🔍  Earnings Screener",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("<div style='color:#4a5568;font-size:0.72rem;'>Data via Yahoo Finance · Not financial advice</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — DCF VALUATION
# ══════════════════════════════════════════════════════════════
if page == "📊  DCF Valuation":

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_dcf_data(ticker):
        tk = yf.Ticker(ticker)
        info = tk.info
        cashflow = tk.cashflow
        balance  = tk.balance_sheet
        if not info or not info.get("regularMarketPrice"):
            raise ValueError(f"No data found for '{ticker}'.")
        fcf_history, fcf_years = [], []
        for row_op, row_cap in [("Operating Cash Flow","Capital Expenditure"),
                                  ("Total Cash From Operating Activities","Capital Expenditures")]:
            try:
                op  = cashflow.loc[row_op]
                cap = cashflow.loc[row_cap]
                fcf_raw = (op + cap).dropna()
                fcf_history = fcf_raw.values[::-1].tolist()
                fcf_years   = [str(d.year) for d in fcf_raw.index[::-1]]
                break
            except KeyError: continue
        try:    total_debt = float(balance.loc["Total Debt"].iloc[0])
        except: total_debt = float(info.get("totalDebt",0) or 0)
        try:    cash = float(balance.loc["Cash And Cash Equivalents"].iloc[0])
        except: cash = float(info.get("totalCash",0) or 0)
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or info.get("floatShares") or 1
        return {
            "ticker": ticker.upper(), "name": info.get("longName", ticker.upper()),
            "sector": info.get("sector","N/A"), "industry": info.get("industry","N/A"),
            "currency": info.get("currency","USD"),
            "current_price": info.get("regularMarketPrice") or info.get("currentPrice"),
            "market_cap": info.get("marketCap"), "shares": shares,
            "net_debt": total_debt - cash, "fcf_history": fcf_history, "fcf_years": fcf_years,
            "pe_ratio": info.get("trailingPE"), "forward_pe": info.get("forwardPE"),
            "ev_ebitda": info.get("enterpriseToEbitda"), "beta": info.get("beta"),
            "analyst_target": info.get("targetMeanPrice"),
            "description": info.get("longBusinessSummary",""),
        }

    def estimate_growth(fcf_history, years=5):
        data = [f for f in fcf_history if f and not np.isnan(f)]
        if len(data) < 2: return 0.08
        data = data[-(years+1):]
        s, e, n = data[0], data[-1], len(data)-1
        if s <= 0 or e <= 0:
            pos = [f for f in data if f > 0]
            return ((pos[-1]/pos[0])**(1/(len(pos)-1))-1) if len(pos)>=2 else 0.08
        return max(min((e/s)**(1/n)-1, 0.40), -0.15)

    def run_dcf(base_fcf, gr, wacc, tgr, yrs, net_debt, shares):
        years    = list(range(1, yrs+1))
        proj     = [base_fcf*(1+gr)**y for y in years]
        disc     = [f/(1+wacc)**y for y,f in zip(years,proj)]
        tv       = proj[-1]*(1+tgr)/(wacc-tgr)
        dtv      = tv/(1+wacc)**yrs
        pv       = sum(disc)
        ev       = pv + dtv
        eq       = ev - net_debt
        return {"years":years,"proj_fcf":proj,"disc_fcf":disc,"pv_fcf":pv,
                "terminal_val":tv,"disc_terminal":dtv,"enterprise_value":ev,
                "equity_value":eq,"intrinsic_per_share":eq/shares if shares else 0}

    with st.sidebar:
        st.markdown("<div style='color:#4a5568;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;'>DCF Settings</div>", unsafe_allow_html=True)
        dcf_ticker = st.text_input("Ticker","AAPL",key="dcf_ticker").strip().upper()
        dcf_run    = st.button("▶  Run DCF", type="primary", use_container_width=True, key="dcf_run")
        wacc       = st.slider("WACC (%)", 5.0, 20.0, 10.0, 0.5, key="dcf_wacc") / 100
        tgr        = st.slider("Terminal Growth (%)", 0.5, 5.0, 2.5, 0.25, key="dcf_tgr") / 100
        proj_yrs   = st.slider("Projection Years", 3, 10, 5, 1, key="dcf_yrs")
        mos        = st.slider("Margin of Safety (%)", 0, 40, 20, 5, key="dcf_mos") / 100
        override   = st.checkbox("Override growth rate", key="dcf_override")
        manual_gr  = st.slider("FCF Growth (%)", -10.0, 40.0, 10.0, 0.5, key="dcf_mgr") / 100 if override else None

    st.markdown("<div class='hero'><div class='hero-title'>DCF Valuation</div><div class='hero-sub'>Discounted Cash Flow · Intrinsic Value · Sensitivity Analysis</div></div>", unsafe_allow_html=True)

    if dcf_run or "dcf_data" in st.session_state:
        if dcf_run:
            with st.spinner(f"Fetching {dcf_ticker}…"):
                try:
                    d = fetch_dcf_data(dcf_ticker)
                    st.session_state["dcf_data"] = d
                except Exception as e:
                    st.error(f"❌ {e}"); st.stop()
        d = st.session_state["dcf_data"]
        if not d["fcf_history"]: st.error("No FCF data."); st.stop()

        base_fcf_auto = d["fcf_history"][-1]
        gr = manual_gr if manual_gr is not None else estimate_growth(d["fcf_history"])
        base_fcf = base_fcf_auto if base_fcf_auto > 0 else np.mean([f for f in d["fcf_history"] if f > 0] or [0])
        if base_fcf <= 0: st.error("No positive FCF available."); st.stop()
        if wacc <= tgr: st.error("WACC must exceed terminal growth."); st.stop()

        res = run_dcf(base_fcf, gr, wacc, tgr, proj_yrs, d["net_debt"], d["shares"])
        intrinsic = res["intrinsic_per_share"]
        mos_price = intrinsic*(1-mos)
        price     = d["current_price"]
        upside    = (intrinsic-price)/price

        st.markdown(f"<div class='section-title'>{d['name']} · <span style='color:#4a5568;font-size:0.9rem;'>{d['sector']} · {d['industry']}</span></div>", unsafe_allow_html=True)
        if d["description"]:
            with st.expander("Company Description"):
                st.write(d["description"][:600]+"…" if len(d["description"])>600 else d["description"])

        if upside > 0.25:
            st.markdown(f"<div class='verdict-undervalued'>✔  Potentially Undervalued — {upside*100:.1f}% above current price</div>", unsafe_allow_html=True)
        elif upside < -0.25:
            st.markdown(f"<div class='verdict-overvalued'>✘  Potentially Overvalued — {abs(upside)*100:.1f}% below current price</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='verdict-fair'>~  Fairly Valued ({upside*100:+.1f}%)</div>", unsafe_allow_html=True)

        c1,c2,c3,c4,c5 = st.columns(5)
        uc = "green" if upside>0 else "red"
        for col, lbl, val, cls in [
            (c1,"Current Price",fmt_p(price),""),
            (c2,"Intrinsic Value",fmt_p(intrinsic),"green"),
            (c3,f"MoS Price ({mos*100:.0f}%)",fmt_p(mos_price),""),
            (c4,"Upside/Downside",f"{upside*100:+.1f}%",uc),
            (c5,"Market Cap",fmt_b(d["market_cap"]),""),
        ]:
            col.markdown(f"<div class='metric-card'><div class='metric-label'>{lbl}</div><div class='metric-value {cls}'>{val}</div></div>", unsafe_allow_html=True)

        d1,d2,d3,d4 = st.columns(4)
        for col, lbl, val in [
            (d1,"P/E (trailing)", f"{d['pe_ratio']:.1f}×" if d['pe_ratio'] else "N/A"),
            (d2,"EV/EBITDA", f"{d['ev_ebitda']:.1f}×" if d['ev_ebitda'] else "N/A"),
            (d3,"Beta", f"{d['beta']:.2f}" if d['beta'] else "N/A"),
            (d4,"Analyst Target", fmt_p(d['analyst_target'])),
        ]:
            col.markdown(f"<div class='metric-card'><div class='metric-label'>{lbl}</div><div class='metric-value'>{val}</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Cash Flow Analysis</div>", unsafe_allow_html=True)
        ch1,ch2 = st.columns(2)
        with ch1:
            colors = ["#00d4aa" if f>0 else "#ff6b6b" for f in d["fcf_history"]]
            fig = go.Figure(go.Bar(x=d["fcf_years"],y=[f/1e9 for f in d["fcf_history"]],
                marker_color=colors,text=[f"${f/1e9:.2f}B" for f in d["fcf_history"]],textposition="outside"))
            fig.update_layout(**CHART_THEME,title="Historical FCF (B)",title_font=dict(size=14,color="#e8eaf0"))
            st.plotly_chart(fig, use_container_width=True)
        with ch2:
            yrs_lbl = [f"Year {y}" for y in res["years"]]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="Projected",x=yrs_lbl,y=[f/1e9 for f in res["proj_fcf"]],marker_color="#2563eb",opacity=0.7))
            fig2.add_trace(go.Bar(name="Discounted",x=yrs_lbl,y=[f/1e9 for f in res["disc_fcf"]],marker_color="#00d4aa"))
            fig2.update_layout(**CHART_THEME,barmode="group",title="Projected vs Discounted FCF (B)",title_font=dict(size=14,color="#e8eaf0"))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<div class='section-title'>Sensitivity Analysis (Intrinsic Value per Share)</div>", unsafe_allow_html=True)
        wr = [max(wacc-0.02,0.01),max(wacc-0.01,0.01),wacc,wacc+0.01,wacc+0.02]
        tr = [t for t in [0.010,0.015,0.020,0.025,0.030,0.035,0.040] if t < wacc]
        mat = []
        for t in tr:
            row=[]
            for w in wr:
                if w<=t: row.append(np.nan)
                else:
                    r2=run_dcf(base_fcf,gr,w,t,proj_yrs,d["net_debt"],d["shares"])
                    row.append(r2["intrinsic_per_share"])
            mat.append(row)
        mat = np.array(mat)
        fig3 = go.Figure(go.Heatmap(
            z=mat, x=[f"{w*100:.1f}%" for w in wr], y=[f"{t*100:.1f}%" for t in tr],
            colorscale=[[0,"#ff6b6b"],[0.5,"#f6ad55"],[1,"#00d4aa"]], zmid=price,
            text=[[f"${v:.2f}" if not np.isnan(v) else "N/A" for v in row] for row in mat],
            texttemplate="%{text}", textfont=dict(size=11),
            colorbar=dict(title="$/share",tickfont=dict(color="#8892a4"))))
        fig3.update_layout(**CHART_THEME,title="WACC × Terminal Growth",title_font=dict(size=14,color="#e8eaf0"),
                           xaxis_title="WACC →",yaxis_title="Terminal Growth →")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("🟢 Green = above current price · 🔴 Red = below current price")

        with st.expander("📋 Full DCF Detail"):
            cf_df = pd.DataFrame({"Year":[f"Year {y}" for y in res["years"]],
                "Projected FCF":[fmt_m(f) for f in res["proj_fcf"]],
                "Discounted FCF":[fmt_m(f) for f in res["disc_fcf"]]})
            st.dataframe(cf_df, use_container_width=True, hide_index=True)
            sum_df = pd.DataFrame({"Component":["PV of FCFs","Terminal Value (PV)","Enterprise Value","Net Debt","Equity Value","Intrinsic/Share"],
                "Value":[fmt_m(res["pv_fcf"]),fmt_m(res["disc_terminal"]),fmt_m(res["enterprise_value"]),fmt_m(d["net_debt"]),fmt_m(res["equity_value"]),fmt_p(intrinsic)]})
            st.dataframe(sum_df, use_container_width=True, hide_index=True)

        st.markdown("<div class='disclaimer'>⚠ For educational purposes only. Not financial advice.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center;padding:4rem 2rem;color:#4a5568;'><div style='font-size:3rem;'>📊</div><div style='font-family:DM Serif Display,serif;font-size:1.6rem;color:#8892a4;margin:1rem 0;'>Enter a ticker and run your valuation</div></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — COMPARABLE COMPANIES (COMPS)
# ══════════════════════════════════════════════════════════════
elif page == "🏢  Comparable Companies":

    SECTOR_TICKERS = {
        "Technology (US)":    ["AAPL","MSFT","GOOGL","META","NVDA","ORCL","CRM","ADBE","INTC","AMD","QCOM","TXN","AVGO","NOW","SNOW"],
        "Banking (US)":       ["JPM","BAC","WFC","GS","MS","C","USB","PNC","TFC","COF","SCHW","BK","STT","AXP","DFS"],
        "Healthcare (US)":    ["JNJ","UNH","PFE","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN","GILD","ISRG","MDT","BSX","SYK"],
        "Consumer Retail":    ["AMZN","WMT","TGT","COST","HD","LOW","NKE","SBUX","MCD","YUM","LULU","ROST","TJX","DG","DLTR"],
        "Energy (US)":        ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","PXD","HAL","BKR","DVN","FANG","HES"],
        "UK Large Cap":       ["SHEL.L","BP.L","HSBA.L","ULVR.L","AZN.L","GSK.L","DGE.L","BATS.L","RIO.L","BHP.L","VOD.L","LLOY.L","BARC.L","NWG.L","PRU.L"],
    }

    with st.sidebar:
        st.markdown("<div style='color:#4a5568;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;'>Comps Settings</div>", unsafe_allow_html=True)
        sector_choice = st.selectbox("Sector", list(SECTOR_TICKERS.keys()), key="comps_sector")
        custom_input  = st.text_input("Or enter tickers (comma-separated)", key="comps_custom", placeholder="e.g. AAPL, MSFT, GOOGL")
        comps_run     = st.button("▶  Run Comps", type="primary", use_container_width=True, key="comps_run")

    st.markdown("<div class='hero'><div class='hero-title'>Comparable Company Analysis</div><div class='hero-sub'>EV/EBITDA · P/E · Debt/EBITDA · FCF Yield · Revenue Multiples</div></div>", unsafe_allow_html=True)

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_comps(tickers):
        rows = []
        for t in tickers:
            try:
                info = yf.Ticker(t).info
                name        = info.get("longName", t)[:28]
                price       = safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
                mktcap      = safe_float(info.get("marketCap"))
                ev          = safe_float(info.get("enterpriseValue"))
                ebitda      = safe_float(info.get("ebitda"))
                rev         = safe_float(info.get("totalRevenue"))
                pe          = safe_float(info.get("trailingPE"))
                fpe         = safe_float(info.get("forwardPE"))
                debt        = safe_float(info.get("totalDebt"))
                fcf         = safe_float(info.get("freeCashflow"))
                ev_ebitda   = ev/ebitda   if ev   and ebitda and ebitda>0 else np.nan
                ev_rev      = ev/rev      if ev   and rev    and rev>0    else np.nan
                debt_ebitda = debt/ebitda if debt and ebitda and ebitda>0 else np.nan
                fcf_yield   = fcf/mktcap  if fcf  and mktcap and mktcap>0 else np.nan
                rows.append({"Ticker":t,"Company":name,"Price":price,
                              "Mkt Cap (B)":mktcap/1e9 if mktcap else np.nan,
                              "EV/EBITDA":ev_ebitda,"EV/Revenue":ev_rev,
                              "P/E (trail)":pe,"P/E (fwd)":fpe,
                              "Debt/EBITDA":debt_ebitda,"FCF Yield":fcf_yield})
            except: continue
        return pd.DataFrame(rows)

    if comps_run or "comps_df" in st.session_state:
        if comps_run:
            tickers = [t.strip().upper() for t in custom_input.split(",")] if custom_input.strip() \
                      else SECTOR_TICKERS[sector_choice]
            with st.spinner(f"Fetching {len(tickers)} companies…"):
                df = fetch_comps(tuple(tickers))
                st.session_state["comps_df"] = df
                st.session_state["comps_sector_name"] = sector_choice if not custom_input.strip() else "Custom"
        df = st.session_state["comps_df"]

        st.markdown(f"<div class='section-title'>Comps Table — {st.session_state.get('comps_sector_name','')}</div>", unsafe_allow_html=True)

        display = df.copy()
        display["Price"]        = display["Price"].apply(lambda x: f"${x:,.2f}" if not np.isnan(x) else "N/A")
        display["Mkt Cap (B)"]  = display["Mkt Cap (B)"].apply(lambda x: f"${x:,.1f}B" if not np.isnan(x) else "N/A")
        for col in ["EV/EBITDA","EV/Revenue","P/E (trail)","P/E (fwd)","Debt/EBITDA"]:
            display[col] = display[col].apply(lambda x: f"{x:.1f}×" if not np.isnan(x) else "N/A")
        display["FCF Yield"] = display["FCF Yield"].apply(lambda x: f"{x*100:.1f}%" if not np.isnan(x) else "N/A")
        st.dataframe(display, use_container_width=True, hide_index=True)

        st.markdown("<div class='section-title'>Sector Benchmarks</div>", unsafe_allow_html=True)
        num_cols = ["EV/EBITDA","EV/Revenue","P/E (trail)","P/E (fwd)","Debt/EBITDA","FCF Yield"]
        medians  = {c: df[c].median() for c in num_cols}
        means    = {c: df[c].mean()   for c in num_cols}
        bench_df = pd.DataFrame({"Metric":num_cols,
            "Median":[f"{medians[c]:.1f}×" if c!="FCF Yield" else f"{medians[c]*100:.1f}%" for c in num_cols],
            "Mean":  [f"{means[c]:.1f}×"   if c!="FCF Yield" else f"{means[c]*100:.1f}%"   for c in num_cols]})
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

        st.markdown("<div class='section-title'>Visual Comparison</div>", unsafe_allow_html=True)
        col1,col2 = st.columns(2)
        with col1:
            nd = df.dropna(subset=["EV/EBITDA"])
            fig = go.Figure(go.Bar(x=nd["Ticker"],y=nd["EV/EBITDA"],
                marker_color=["#00d4aa" if v<medians["EV/EBITDA"] else "#2563eb" for v in nd["EV/EBITDA"]],
                text=[f"{v:.1f}×" for v in nd["EV/EBITDA"]],textposition="outside"))
            fig.add_hline(y=medians["EV/EBITDA"],line_dash="dash",line_color="#f6ad55",
                          annotation_text=f"Median {medians['EV/EBITDA']:.1f}×")
            fig.update_layout(**CHART_THEME,title="EV/EBITDA",title_font=dict(size=14,color="#e8eaf0"))
            st.plotly_chart(fig,use_container_width=True)
        with col2:
            pd2 = df.dropna(subset=["P/E (trail)"])
            pd2 = pd2[pd2["P/E (trail)"]<100]
            fig2 = go.Figure(go.Bar(x=pd2["Ticker"],y=pd2["P/E (trail)"],
                marker_color=["#00d4aa" if v<medians["P/E (trail)"] else "#ff6b6b" for v in pd2["P/E (trail)"]],
                text=[f"{v:.1f}×" for v in pd2["P/E (trail)"]],textposition="outside"))
            fig2.add_hline(y=medians["P/E (trail)"],line_dash="dash",line_color="#f6ad55",
                           annotation_text=f"Median {medians['P/E (trail)']:.1f}×")
            fig2.update_layout(**CHART_THEME,title="P/E Ratio (Trailing)",title_font=dict(size=14,color="#e8eaf0"))
            st.plotly_chart(fig2,use_container_width=True)

        col3,col4 = st.columns(2)
        with col3:
            dd = df.dropna(subset=["Debt/EBITDA"])
            fig3 = go.Figure(go.Bar(x=dd["Ticker"],y=dd["Debt/EBITDA"],
                marker_color=["#ff6b6b" if v>3.0 else "#00d4aa" for v in dd["Debt/EBITDA"]],
                text=[f"{v:.1f}×" for v in dd["Debt/EBITDA"]],textposition="outside"))
            fig3.add_hline(y=3.0,line_dash="dash",line_color="#f6ad55",annotation_text="3.0× threshold")
            fig3.update_layout(**CHART_THEME,title="Debt/EBITDA (🔴 >3×)",title_font=dict(size=14,color="#e8eaf0"))
            st.plotly_chart(fig3,use_container_width=True)
        with col4:
            fd = df.dropna(subset=["FCF Yield"])
            fig4 = go.Figure(go.Bar(x=fd["Ticker"],y=fd["FCF Yield"]*100,
                marker_color=["#00d4aa" if v>5.0 else "#2563eb" for v in fd["FCF Yield"]*100],
                text=[f"{v:.1f}%" for v in fd["FCF Yield"]*100],textposition="outside"))
            fig4.add_hline(y=5.0,line_dash="dash",line_color="#f6ad55",annotation_text="5% threshold")
            fig4.update_layout(**CHART_THEME,title="FCF Yield (🟢 >5%)",title_font=dict(size=14,color="#e8eaf0"))
            st.plotly_chart(fig4,use_container_width=True)

        st.markdown("<div class='disclaimer'>⚠ For educational purposes only. Not financial advice.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center;padding:4rem 2rem;color:#4a5568;'><div style='font-size:3rem;'>🏢</div><div style='font-family:DM Serif Display,serif;font-size:1.6rem;color:#8892a4;margin:1rem 0;'>Select a sector and run Comps</div></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — 3-STATEMENT MODEL
# ══════════════════════════════════════════════════════════════
elif page == "📑  3-Statement Model":

    with st.sidebar:
        st.markdown("<div style='color:#4a5568;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;'>3-Statement Settings</div>", unsafe_allow_html=True)
        stmt_ticker = st.text_input("Ticker","AAPL",key="stmt_ticker").strip().upper()
        stmt_run    = st.button("▶  Load Statements", type="primary", use_container_width=True, key="stmt_run")
        st.divider()
        st.markdown("<div style='color:#4a5568;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;'>Forecast Assumptions</div>", unsafe_allow_html=True)
        rev_gr    = st.slider("Revenue Growth (%)", -10.0, 40.0, 8.0, 0.5, key="rev_gr")   / 100
        op_margin = st.slider("Operating Margin (%)", 1.0, 50.0, 20.0, 0.5, key="op_margin") / 100
        tax_rate  = st.slider("Tax Rate (%)", 5.0, 35.0, 21.0, 0.5, key="tax_rate")         / 100
        capex_pct = st.slider("Capex % of Revenue", 1.0, 20.0, 5.0, 0.5, key="capex_pct")  / 100
        da_pct    = st.slider("D&A % of Revenue", 1.0, 15.0, 4.0, 0.5, key="da_pct")        / 100
        nwc_pct   = st.slider("NWC Change % of Rev", -5.0, 5.0, 1.0, 0.5, key="nwc_pct")   / 100

    st.markdown("<div class='hero'><div class='hero-title'>3-Statement Model</div><div class='hero-sub'>Income Statement · Balance Sheet · Cash Flow · 5-Year Forecast</div></div>", unsafe_allow_html=True)

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_statements(ticker):
        tk   = yf.Ticker(ticker)
        info = tk.info
        inc  = tk.income_stmt
        bal  = tk.balance_sheet
        cf   = tk.cashflow
        return info, inc, bal, cf

    if stmt_run or "stmt_data" in st.session_state:
        if stmt_run:
            with st.spinner(f"Loading {stmt_ticker}…"):
                try:
                    info, inc, bal, cf = fetch_statements(stmt_ticker)
                    st.session_state["stmt_data"] = (info, inc, bal, cf, stmt_ticker)
                except Exception as e:
                    st.error(f"❌ {e}"); st.stop()
        info, inc, bal, cf, loaded_ticker = st.session_state["stmt_data"]
        name = info.get("longName", loaded_ticker)

        st.markdown(f"<div class='section-title'>{name} — Historical Statements</div>", unsafe_allow_html=True)

        try:
            inc_d = inc.copy()
            inc_d.columns = [str(c.year) for c in inc_d.columns]
            inc_d = (inc_d/1e9).round(2)
            key_rows = ["Total Revenue","Gross Profit","Operating Income","Net Income","EBITDA"]
            inc_f = inc_d.loc[[r for r in key_rows if r in inc_d.index]]
            st.markdown("**Income Statement (USD Billions)**")
            st.dataframe(inc_f, use_container_width=True)
        except: st.info("Income statement not available.")

        try:
            cf_d = cf.copy()
            cf_d.columns = [str(c.year) for c in cf_d.columns]
            cf_d = (cf_d/1e9).round(2)
            cf_rows = ["Operating Cash Flow","Capital Expenditure","Total Cash From Operating Activities","Capital Expenditures"]
            cf_f = cf_d.loc[[r for r in cf_rows if r in cf_d.index]]
            st.markdown("**Cash Flow Statement (USD Billions)**")
            st.dataframe(cf_f, use_container_width=True)
        except: st.info("Cash flow not available.")

        try:
            bal_d = bal.copy()
            bal_d.columns = [str(c.year) for c in bal_d.columns]
            bal_d = (bal_d/1e9).round(2)
            bal_rows = ["Total Assets","Total Liabilities Net Minority Interest","Stockholders Equity","Total Debt","Cash And Cash Equivalents"]
            bal_f = bal_d.loc[[r for r in bal_rows if r in bal_d.index]]
            st.markdown("**Balance Sheet (USD Billions)**")
            st.dataframe(bal_f, use_container_width=True)
        except: st.info("Balance sheet not available.")

        # ── 5-Year Forecast ──────────────────────────────────────────────────
        st.markdown("<div class='section-title'>5-Year Forecast</div>", unsafe_allow_html=True)
        try:
            base_rev = None
            for row_name in ["Total Revenue","Revenue"]:
                try: base_rev = float(inc.loc[row_name].iloc[0]); break
                except: continue
            if base_rev is None or np.isnan(base_rev): raise ValueError
        except:
            st.error("Could not extract base revenue."); st.stop()

        forecast_years = [f"Y+{i}" for i in range(1,6)]
        revenues, op_incomes, ebitdas, net_incomes, fcfs = [], [], [], [], []
        rev = base_rev
        for _ in range(5):
            rev       *= (1+rev_gr)
            op_inc     = rev*op_margin
            ebitda_val = op_inc + rev*da_pct
            nopat      = op_inc*(1-tax_rate)
            capex      = rev*capex_pct
            da         = rev*da_pct
            dnwc       = rev*nwc_pct
            fcf        = nopat + da - capex - dnwc
            revenues.append(rev/1e9); op_incomes.append(op_inc/1e9)
            ebitdas.append(ebitda_val/1e9); net_incomes.append(nopat/1e9); fcfs.append(fcf/1e9)

        forecast_df = pd.DataFrame({
            "Year":forecast_years,
            "Revenue ($B)":[f"${v:.2f}B" for v in revenues],
            "Op. Income ($B)":[f"${v:.2f}B" for v in op_incomes],
            "EBITDA ($B)":[f"${v:.2f}B" for v in ebitdas],
            "Net Income ($B)":[f"${v:.2f}B" for v in net_incomes],
            "FCF ($B)":[f"${v:.2f}B" for v in fcfs],
        })
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Revenue",x=forecast_years,y=revenues,marker_color="#2563eb",opacity=0.6))
        fig.add_trace(go.Bar(name="EBITDA",x=forecast_years,y=ebitdas,marker_color="#00d4aa"))
        fig.add_trace(go.Bar(name="FCF",x=forecast_years,y=fcfs,marker_color="#f6ad55"))
        fig.update_layout(**CHART_THEME,barmode="group",title="5-Year Forecast (USD Billions)",
                          title_font=dict(size=14,color="#e8eaf0"))
        st.plotly_chart(fig,use_container_width=True)

        # ── Sensitivity Heatmap ──────────────────────────────────────────────
        st.markdown("<div class='section-title'>Sensitivity: Revenue Growth × Operating Margin → Year 5 FCF ($B)</div>", unsafe_allow_html=True)
        rev_range    = [r/100 for r in np.arange(max(rev_gr*100-6,1), rev_gr*100+8, 2)]
        margin_range = [m/100 for m in np.arange(max(op_margin*100-8,1), op_margin*100+10, 2)]
        sens = []
        for mg in margin_range:
            row = []
            for rg in rev_range:
                r = base_rev
                for _ in range(5): r *= (1+rg)
                f = r*mg*(1-tax_rate) + r*da_pct - r*capex_pct - r*nwc_pct
                row.append(f/1e9)
            sens.append(row)
        sens = np.array(sens)
        fig_s = go.Figure(go.Heatmap(
            z=sens, x=[f"{r*100:.1f}%" for r in rev_range], y=[f"{m*100:.1f}%" for m in margin_range],
            colorscale=[[0,"#ff6b6b"],[0.5,"#f6ad55"],[1,"#00d4aa"]],
            text=[[f"${v:.1f}B" for v in row] for row in sens],
            texttemplate="%{text}", textfont=dict(size=11),
            colorbar=dict(title="FCF $B",tickfont=dict(color="#8892a4"))))
        fig_s.update_layout(**CHART_THEME,
            title="Year 5 FCF — Revenue Growth × Operating Margin",
            title_font=dict(size=14,color="#e8eaf0"),
            xaxis_title="Revenue Growth Rate →", yaxis_title="Operating Margin →")
        st.plotly_chart(fig_s,use_container_width=True)
        st.caption("How Year 5 Free Cash Flow changes across different growth and margin scenarios")

        st.markdown("<div class='disclaimer'>⚠ For educational purposes only. Not financial advice.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='text-align:center;padding:4rem 2rem;color:#4a5568;'><div style='font-size:3rem;'>📑</div><div style='font-family:DM Serif Display,serif;font-size:1.6rem;color:#8892a4;margin:1rem 0;'>Enter a ticker to load financial statements</div></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — INSTITUTIONAL EARNINGS SCREENER
# ══════════════════════════════════════════════════════════════
elif page == "🔍  Earnings Screener":

    SP500_SAMPLE = [
        "AAPL","MSFT","GOOGL","AMZN","NVDA","META","BRK-B","LLY","AVGO","JPM",
        "TSLA","UNH","XOM","V","MA","JNJ","PG","HD","COST","MRK",
        "ABBV","CVX","KO","PEP","ADBE","WMT","CRM","BAC","TMO","ORCL",
        "MCD","CSCO","ACN","ABT","NKE","LIN","DHR","NEE","PM","IBM",
        "RTX","QCOM","T","LOW","UPS","GE","CAT","SPGI","MS","BLK",
        "INTU","ISRG","AMGN","SYK","GS","AXP","DE","MDLZ","ADI","REGN",
        "PLD","CI","TJX","MMC","VRTX","CB","HUM","BSX","NOW","ZTS",
        "C","MO","GILD","EOG","COP","SLB","USB","WFC","PNC","TGT",
        "F","GM","BA","MMM","DIS","NFLX","PYPL","INTC","AMD","TXN",
    ]

    with st.sidebar:
        st.markdown("<div style='color:#4a5568;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;'>Screener Filters</div>", unsafe_allow_html=True)
        max_debt_ebitda  = st.slider("Max Debt/EBITDA", 0.5, 8.0, 3.0, 0.5, key="scr_debt")
        min_fcf_yield    = st.slider("Min FCF Yield (%)", 0.0, 15.0, 5.0, 0.5, key="scr_fcf") / 100
        max_pe           = st.slider("Max P/E", 5.0, 60.0, 25.0, 1.0, key="scr_pe")
        min_market_cap_b = st.slider("Min Market Cap ($B)", 1.0, 100.0, 10.0, 1.0, key="scr_cap")
        sample_size      = st.slider("Companies to scan", 20, 90, 50, 10, key="scr_size")
        scr_run          = st.button("▶  Run Screener", type="primary", use_container_width=True, key="scr_run")

    st.markdown("<div class='hero'><div class='hero-title'>Institutional Earnings Screener</div><div class='hero-sub'>Value Criteria · FCF Yield · Debt/EBITDA · P/E Filter · Sector Breakdown</div></div>", unsafe_allow_html=True)

    if scr_run:
        tickers_to_scan = SP500_SAMPLE[:sample_size]
        progress = st.progress(0, text="Scanning companies…")
        results_list = []
        for i, t in enumerate(tickers_to_scan):
            progress.progress((i+1)/len(tickers_to_scan), text=f"Scanning {t}…")
            try:
                info    = yf.Ticker(t).info
                mktcap  = safe_float(info.get("marketCap"))
                if np.isnan(mktcap) or mktcap < min_market_cap_b*1e9: continue
                ebitda  = safe_float(info.get("ebitda"))
                debt    = safe_float(info.get("totalDebt"))
                fcf     = safe_float(info.get("freeCashflow"))
                pe      = safe_float(info.get("trailingPE"))
                ev      = safe_float(info.get("enterpriseValue"))
                price   = safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
                sector  = info.get("sector","N/A")
                name    = info.get("longName",t)[:30]
                debt_ebitda = debt/ebitda if debt and ebitda and ebitda>0 else np.nan
                fcf_yield   = fcf/mktcap  if fcf  and mktcap and mktcap>0 else np.nan
                ev_ebitda   = ev/ebitda   if ev   and ebitda and ebitda>0 else np.nan
                passes = (
                    not np.isnan(fcf_yield) and not np.isnan(debt_ebitda) and
                    debt_ebitda <= max_debt_ebitda and
                    fcf_yield   >= min_fcf_yield   and
                    (np.isnan(pe) or pe <= max_pe)
                )
                if passes:
                    results_list.append({"Ticker":t,"Company":name,"Sector":sector,
                                         "Price":price,"Mkt Cap ($B)":mktcap/1e9,
                                         "P/E":pe,"EV/EBITDA":ev_ebitda,
                                         "Debt/EBITDA":debt_ebitda,"FCF Yield":fcf_yield})
            except: continue
        progress.empty()
        st.session_state["scr_results"] = pd.DataFrame(results_list)
        st.session_state["scr_params"]  = (max_debt_ebitda, min_fcf_yield, max_pe, min_market_cap_b, sample_size)

    if "scr_results" in st.session_state:
        df_res = st.session_state["scr_results"]
        params = st.session_state.get("scr_params",(max_debt_ebitda,min_fcf_yield,max_pe,min_market_cap_b,sample_size))

        c1,c2,c3,c4 = st.columns(4)
        for col,lbl,val in [
            (c1,"Companies Passed",str(len(df_res))),
            (c2,"Max Debt/EBITDA",f"{params[0]:.1f}×"),
            (c3,"Min FCF Yield",f"{params[1]*100:.1f}%"),
            (c4,"Max P/E",f"{params[2]:.0f}×"),
        ]:
            col.markdown(f"<div class='metric-card'><div class='metric-label'>{lbl}</div><div class='metric-value'>{val}</div></div>",unsafe_allow_html=True)

        if df_res.empty:
            st.warning("No companies passed all filters. Try relaxing the criteria.")
        else:
            st.markdown(f"<div class='section-title'>Results — {len(df_res)} companies passed</div>",unsafe_allow_html=True)
            display = df_res.copy().sort_values("FCF Yield",ascending=False)
            display["Price"]        = display["Price"].apply(lambda x: f"${x:,.2f}" if not np.isnan(x) else "N/A")
            display["Mkt Cap ($B)"] = display["Mkt Cap ($B)"].apply(lambda x: f"${x:,.1f}B" if not np.isnan(x) else "N/A")
            display["P/E"]          = display["P/E"].apply(lambda x: f"{x:.1f}×" if not np.isnan(x) else "N/A")
            display["EV/EBITDA"]    = display["EV/EBITDA"].apply(lambda x: f"{x:.1f}×" if not np.isnan(x) else "N/A")
            display["Debt/EBITDA"]  = display["Debt/EBITDA"].apply(lambda x: f"{x:.1f}×" if not np.isnan(x) else "N/A")
            display["FCF Yield"]    = display["FCF Yield"].apply(lambda x: f"{x*100:.1f}%" if not np.isnan(x) else "N/A")
            st.dataframe(display,use_container_width=True,hide_index=True)

            st.markdown("<div class='section-title'>Visual Analysis</div>",unsafe_allow_html=True)
            plot_df = df_res.sort_values("FCF Yield",ascending=False).head(20)
            ch1,ch2 = st.columns(2)
            with ch1:
                fig = go.Figure(go.Bar(x=plot_df["Ticker"],y=plot_df["FCF Yield"]*100,
                    marker_color="#00d4aa",
                    text=[f"{v:.1f}%" for v in plot_df["FCF Yield"]*100],textposition="outside"))
                fig.add_hline(y=params[1]*100,line_dash="dash",line_color="#f6ad55",
                              annotation_text=f"Min {params[1]*100:.1f}%")
                fig.update_layout(**CHART_THEME,title="FCF Yield (%)",title_font=dict(size=14,color="#e8eaf0"))
                st.plotly_chart(fig,use_container_width=True)
            with ch2:
                de_df = df_res.dropna(subset=["Debt/EBITDA"]).sort_values("Debt/EBITDA")
                fig2 = go.Figure(go.Bar(x=de_df["Ticker"],y=de_df["Debt/EBITDA"],
                    marker_color=["#00d4aa" if v<2.0 else "#f6ad55" for v in de_df["Debt/EBITDA"]],
                    text=[f"{v:.1f}×" for v in de_df["Debt/EBITDA"]],textposition="outside"))
                fig2.add_hline(y=params[0],line_dash="dash",line_color="#ff6b6b",
                               annotation_text=f"Max {params[0]:.1f}×")
                fig2.update_layout(**CHART_THEME,title="Debt/EBITDA",title_font=dict(size=14,color="#e8eaf0"))
                st.plotly_chart(fig2,use_container_width=True)

            st.markdown("<div class='section-title'>Sector Breakdown</div>",unsafe_allow_html=True)
            sc = df_res["Sector"].value_counts().reset_index()
            sc.columns = ["Sector","Count"]
            fig3 = go.Figure(go.Pie(labels=sc["Sector"],values=sc["Count"],
                marker=dict(colors=["#00d4aa","#2563eb","#f6ad55","#ff6b6b","#8892a4",
                                    "#a78bfa","#34d399","#fb923c","#60a5fa","#f472b6"]),
                textfont=dict(color="#e8eaf0")))
            fig3.update_layout(**CHART_THEME,title="Screened Companies by Sector",
                               title_font=dict(size=14,color="#e8eaf0"))
            st.plotly_chart(fig3,use_container_width=True)
            st.info("💡 Take any ticker from the results above and run a full DCF on it using the **📊 DCF Valuation** page.")

        st.markdown("<div class='disclaimer'>⚠ For educational purposes only. Not financial advice.</div>",unsafe_allow_html=True)
    elif not scr_run:
        st.markdown("""
        <div style='text-align:center;padding:4rem 2rem;color:#4a5568;'>
            <div style='font-size:3rem;'>🔍</div>
            <div style='font-family:DM Serif Display,serif;font-size:1.6rem;color:#8892a4;margin:1rem 0;'>Set your filters and run the screener</div>
            <div style='font-size:0.85rem;line-height:1.8;'>
                Filters S&amp;P 500 companies by <strong>Debt/EBITDA</strong>, <strong>FCF Yield</strong>, and <strong>P/E</strong><br>
                Identifies companies matching a value / private equity investment mandate
            </div>
        </div>""", unsafe_allow_html=True)
