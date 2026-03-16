"""
IB Toolkit — Animated Streamlit App
=====================================
streamlit run main.py
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
#  GLOBAL ANIMATED SKIN
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;1,300;1,400&family=Space+Grotesk:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    background: #050810 !important;
    color: #e8eaf0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stAppViewContainer"] > .main { background: #050810 !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }

/* ── Particle canvas background ── */
#particle-canvas {
    position: fixed; top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none; z-index: 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(10,13,26,0.95) !important;
    border-right: 1px solid rgba(0,212,170,0.12) !important;
    backdrop-filter: blur(20px);
}
section[data-testid="stSidebar"] * { color: #e8eaf0 !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #8892a4 !important;
    padding: 0.6rem 0 !important;
    transition: color 0.3s !important;
}
section[data-testid="stSidebar"] .stRadio label:hover { color: #00d4aa !important; }
section[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background: rgba(0,212,170,0.05) !important;
    border: 1px solid rgba(0,212,170,0.2) !important;
    border-radius: 0 !important;
    color: #e8eaf0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
section[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
    border-color: #00d4aa !important;
    box-shadow: 0 0 0 1px rgba(0,212,170,0.3) !important;
}
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #00d4aa !important;
}
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-testid="stSliderTrackFill"] {
    background: #00d4aa !important;
}
section[data-testid="stSidebar"] button[kind="primary"] {
    background: transparent !important;
    border: 1px solid #00d4aa !important;
    color: #00d4aa !important;
    border-radius: 0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    transition: all 0.3s !important;
}
section[data-testid="stSidebar"] button[kind="primary"]:hover {
    background: rgba(0,212,170,0.08) !important;
    box-shadow: 0 0 20px rgba(0,212,170,0.15) !important;
}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
    background: rgba(0,212,170,0.05) !important;
    border: 1px solid rgba(0,212,170,0.2) !important;
    border-radius: 0 !important;
}

/* ── Main content area ── */
.main .block-container {
    padding-top: 2rem !important;
    position: relative; z-index: 10;
}

/* ── Hero block ── */
.ib-hero {
    position: relative;
    padding: 3.5rem 3rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(0,212,170,0.12);
    overflow: hidden;
    animation: fadeUp 0.8s ease forwards;
}
.ib-hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(135deg, rgba(0,212,170,0.03) 0%, transparent 60%);
    pointer-events: none;
}
.ib-hero::after {
    content: '';
    position: absolute; top: -1px; left: 0;
    width: 80px; height: 2px;
    background: #00d4aa;
}
.ib-hero-eyebrow {
    font-size: 0.68rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #00d4aa;
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.75rem;
}
.ib-hero-eyebrow::after {
    content: '';
    flex: 1; height: 1px; max-width: 60px;
    background: linear-gradient(to right, rgba(0,212,170,0.5), transparent);
}
.ib-hero-title {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 300;
    line-height: 1.05;
    color: #e8eaf0;
    margin-bottom: 0.75rem;
}
.ib-hero-title em {
    font-style: italic;
    color: #00d4aa;
}
.ib-hero-sub {
    font-size: 0.82rem;
    color: #4a5568;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Section titles ── */
.ib-section {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.6rem;
    font-weight: 300;
    color: #e8eaf0;
    border-bottom: 1px solid rgba(0,212,170,0.1);
    padding-bottom: 0.6rem;
    margin: 2.5rem 0 1.5rem;
}
.ib-section em { font-style: italic; color: #00d4aa; }

/* ── Metric cards ── */
.ib-card {
    background: rgba(10,13,26,0.8);
    border: 1px solid rgba(0,212,170,0.1);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s, transform 0.3s;
    animation: fadeUp 0.6s ease forwards;
}
.ib-card::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 2px; height: 0;
    background: #00d4aa;
    transition: height 0.4s;
}
.ib-card:hover::before { height: 100%; }
.ib-card:hover { border-color: rgba(0,212,170,0.25); transform: translateY(-2px); }
.ib-card-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #4a5568;
    margin-bottom: 0.4rem;
}
.ib-card-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 300;
    color: #e8eaf0;
    line-height: 1;
}
.ib-card-value.teal { color: #00d4aa; }
.ib-card-value.red  { color: #ff6b6b; }
.ib-card-value.amber{ color: #f6ad55; }

/* ── Verdict banners ── */
.ib-verdict {
    padding: 1.2rem 1.8rem;
    margin: 1.2rem 0;
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.15rem;
    font-weight: 300;
    position: relative;
    animation: fadeUp 0.6s ease forwards;
}
.ib-verdict.under {
    background: linear-gradient(90deg, rgba(0,212,170,0.06), transparent);
    border-left: 2px solid #00d4aa;
    color: #00d4aa;
}
.ib-verdict.over {
    background: linear-gradient(90deg, rgba(255,107,107,0.06), transparent);
    border-left: 2px solid #ff6b6b;
    color: #ff6b6b;
}
.ib-verdict.fair {
    background: linear-gradient(90deg, rgba(246,173,85,0.06), transparent);
    border-left: 2px solid #f6ad55;
    color: #f6ad55;
}

/* ── Disclaimer ── */
.ib-disclaimer {
    border: 1px solid rgba(255,255,255,0.04);
    padding: 1rem 1.5rem;
    color: #2a3044;
    font-size: 0.72rem;
    line-height: 1.7;
    margin-top: 3rem;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,212,170,0.1) !important;
    border-radius: 0 !important;
}
[data-testid="stDataFrame"] th {
    background: rgba(0,212,170,0.06) !important;
    color: #00d4aa !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid rgba(0,212,170,0.15) !important;
}
[data-testid="stDataFrame"] td {
    color: #8892a4 !important;
    font-size: 0.82rem !important;
    border-bottom: 1px solid rgba(255,255,255,0.03) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(0,212,170,0.1) !important;
    border-radius: 0 !important;
    background: rgba(10,13,26,0.6) !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #00d4aa, #00b894) !important;
}

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
}
.pulse { animation: pulse 2s ease infinite; }

/* ── Ticker tape ── */
.ib-tape-outer {
    overflow: hidden;
    border-top: 1px solid rgba(0,212,170,0.08);
    border-bottom: 1px solid rgba(0,212,170,0.08);
    padding: 0.6rem 0;
    margin-bottom: 2rem;
    background: rgba(5,8,16,0.7);
}
.ib-tape {
    display: flex; gap: 3rem;
    white-space: nowrap;
    animation: tape 35s linear infinite;
}
.ib-tape-item {
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4a5568;
}
.ib-tape-item .up { color: #00d4aa; }
.ib-tape-item .dn { color: #ff6b6b; }
@keyframes tape {
    from { transform: translateX(0); }
    to   { transform: translateX(-50%); }
}

/* ── Landing page ── */
.ib-landing {
    min-height: 70vh;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; padding: 4rem 2rem;
    animation: fadeUp 1s ease forwards;
}
.ib-landing-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(3.5rem, 8vw, 7rem);
    font-weight: 300; line-height: 0.95;
    color: #e8eaf0; margin-bottom: 1.5rem;
}
.ib-landing-title em { font-style: italic; color: #00d4aa; display: block; }
.ib-landing-sub {
    font-size: 0.82rem; color: #4a5568;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 3rem;
}
.ib-module-pills {
    display: flex; flex-wrap: wrap; gap: 0.75rem;
    justify-content: center; margin-bottom: 3rem;
}
.ib-pill {
    padding: 0.5rem 1.2rem;
    border: 1px solid rgba(0,212,170,0.2);
    font-size: 0.68rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: #4a5568;
}
.ib-stats-row {
    display: flex; gap: 0; margin-top: 3rem;
    border: 1px solid rgba(0,212,170,0.1);
}
.ib-stat {
    padding: 1.5rem 2.5rem; text-align: center;
    border-right: 1px solid rgba(0,212,170,0.1);
}
.ib-stat:last-child { border-right: none; }
.ib-stat-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem; font-weight: 300; color: #00d4aa; display: block;
}
.ib-stat-lbl {
    font-size: 0.62rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: #4a5568;
    margin-top: 0.25rem; display: block;
}
</style>
""", unsafe_allow_html=True)

# ── Particle canvas (injected once) ──────────────────────────────────────────
import streamlit.components.v1 as components
components.html("""
<canvas id="particle-canvas" style="position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;"></canvas>
<script>
(function(){
  var c = document.getElementById('particle-canvas');
  if (!c) return;
  var ctx = c.getContext('2d');
  function resize(){ c.width = window.innerWidth; c.height = window.innerHeight; }
  resize();
  window.addEventListener('resize', resize);
  var pts = [];
  for(var i=0;i<70;i++){
    pts.push({
      x: Math.random()*c.width, y: Math.random()*c.height,
      vx:(Math.random()-0.5)*0.2, vy:(Math.random()-0.5)*0.2,
      r: Math.random()*1.2+0.3, o: Math.random()*0.25+0.05
    });
  }
  function draw(){
    ctx.clearRect(0,0,c.width,c.height);
    for(var i=0;i<pts.length;i++){
      var p=pts[i];
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0||p.x>c.width)  p.vx*=-1;
      if(p.y<0||p.y>c.height) p.vy*=-1;
      ctx.beginPath();
      ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle='rgba(0,212,170,'+p.o+')';
      ctx.fill();
      for(var j=i+1;j<pts.length;j++){
        var q=pts[j];
        var d=Math.hypot(p.x-q.x,p.y-q.y);
        if(d<110){
          ctx.beginPath();
          ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y);
          ctx.strokeStyle='rgba(0,212,170,'+(0.05*(1-d/110))+')';
          ctx.lineWidth=0.5; ctx.stroke();
        }
      }
    }
    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
""", height=0)

# ── Ticker tape ──────────────────────────────────────────────────────────────
TAPE_HTML = """
<div class="ib-tape-outer">
  <div class="ib-tape">
    <div class="ib-tape-item">AAPL <span class="up">+1.24%</span></div>
    <div class="ib-tape-item">MSFT <span class="dn">-0.38%</span></div>
    <div class="ib-tape-item">NVDA <span class="up">+3.11%</span></div>
    <div class="ib-tape-item">GOOGL <span class="up">+0.87%</span></div>
    <div class="ib-tape-item">TSLA <span class="dn">-1.55%</span></div>
    <div class="ib-tape-item">JPM <span class="up">+0.63%</span></div>
    <div class="ib-tape-item">META <span class="up">+2.04%</span></div>
    <div class="ib-tape-item">XOM <span class="dn">-0.22%</span></div>
    <div class="ib-tape-item">AMZN <span class="up">+1.78%</span></div>
    <div class="ib-tape-item">SHEL.L <span class="dn">-0.19%</span></div>
    <div class="ib-tape-item">AZN.L <span class="up">+1.02%</span></div>
    <div class="ib-tape-item">AVGO <span class="up">+2.33%</span></div>
    <div class="ib-tape-item">AAPL <span class="up">+1.24%</span></div>
    <div class="ib-tape-item">MSFT <span class="dn">-0.38%</span></div>
    <div class="ib-tape-item">NVDA <span class="up">+3.11%</span></div>
    <div class="ib-tape-item">GOOGL <span class="up">+0.87%</span></div>
    <div class="ib-tape-item">TSLA <span class="dn">-1.55%</span></div>
    <div class="ib-tape-item">JPM <span class="up">+0.63%</span></div>
    <div class="ib-tape-item">META <span class="up">+2.04%</span></div>
    <div class="ib-tape-item">XOM <span class="dn">-0.22%</span></div>
  </div>
</div>
"""

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk, sans-serif", color="#8892a4"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"),
    margin=dict(l=10, r=10, t=40, b=10),
)
def safe_float(v):
    try: return float(v)
    except: return np.nan
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

def metric_card(label, value, color=""):
    st.markdown(f"""
    <div class='ib-card'>
        <div class='ib-card-label'>{label}</div>
        <div class='ib-card-value {color}'>{value}</div>
    </div>""", unsafe_allow_html=True)

def section_title(text):
    st.markdown(f"<div class='ib-section'>{text}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR NAV
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 1rem;'>
        <div style='font-family:Cormorant Garamond,serif;font-size:1.6rem;font-weight:300;color:#e8eaf0;'>
            IB <em style='font-style:italic;color:#00d4aa;'>Toolkit</em>
        </div>
        <div style='font-size:0.62rem;letter-spacing:0.2em;text-transform:uppercase;color:#2a3044;margin-top:0.3rem;'>
            Investment Banking Suite
        </div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    page = st.radio("", [
        "📊  DCF Valuation",
        "🏢  Comparable Companies",
        "📑  3-Statement Model",
        "🔍  Earnings Screener",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("<div style='font-size:0.65rem;color:#2a3044;letter-spacing:0.05em;line-height:1.6;'>Data via Yahoo Finance<br>For educational use only</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — DCF VALUATION
# ══════════════════════════════════════════════════════════════
if page == "📊  DCF Valuation":

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_dcf(ticker):
        tk = yf.Ticker(ticker)
        info = tk.info
        cashflow = tk.cashflow
        balance  = tk.balance_sheet
        if not info or not info.get("regularMarketPrice"):
            raise ValueError(f"No data found for '{ticker}'.")
        fcf_history, fcf_years = [], []
        for ro, rc in [("Operating Cash Flow","Capital Expenditure"),
                        ("Total Cash From Operating Activities","Capital Expenditures")]:
            try:
                fcf_raw = (cashflow.loc[ro] + cashflow.loc[rc]).dropna()
                fcf_history = fcf_raw.values[::-1].tolist()
                fcf_years   = [str(d.year) for d in fcf_raw.index[::-1]]
                break
            except KeyError: continue
        try:    debt = float(balance.loc["Total Debt"].iloc[0])
        except: debt = float(info.get("totalDebt",0) or 0)
        try:    cash = float(balance.loc["Cash And Cash Equivalents"].iloc[0])
        except: cash = float(info.get("totalCash",0) or 0)
        shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 1
        return {"ticker":ticker.upper(),"name":info.get("longName",ticker),
                "sector":info.get("sector","N/A"),"industry":info.get("industry","N/A"),
                "current_price":info.get("regularMarketPrice") or info.get("currentPrice"),
                "market_cap":info.get("marketCap"),"shares":shares,
                "net_debt":debt-cash,"fcf_history":fcf_history,"fcf_years":fcf_years,
                "pe_ratio":info.get("trailingPE"),"ev_ebitda":info.get("enterpriseToEbitda"),
                "beta":info.get("beta"),"analyst_target":info.get("targetMeanPrice"),
                "description":info.get("longBusinessSummary","")}

    def est_growth(hist, yrs=5):
        d = [f for f in hist if f and not np.isnan(f)]
        if len(d)<2: return 0.08
        d = d[-(yrs+1):]
        s,e,n = d[0],d[-1],len(d)-1
        if s<=0 or e<=0:
            pos=[f for f in d if f>0]
            return ((pos[-1]/pos[0])**(1/(len(pos)-1))-1) if len(pos)>=2 else 0.08
        return max(min((e/s)**(1/n)-1,0.40),-0.15)

    def run_dcf(bf,gr,wacc,tgr,yrs,nd,sh):
        yl=list(range(1,yrs+1))
        pf=[bf*(1+gr)**y for y in yl]
        df=[f/(1+wacc)**y for y,f in zip(yl,pf)]
        tv=pf[-1]*(1+tgr)/(wacc-tgr)
        dtv=tv/(1+wacc)**yrs
        pv=sum(df); ev=pv+dtv; eq=ev-nd
        return {"years":yl,"proj_fcf":pf,"disc_fcf":df,"pv_fcf":pv,
                "terminal_val":tv,"disc_terminal":dtv,"enterprise_value":ev,
                "equity_value":eq,"intrinsic_per_share":eq/sh if sh else 0}

    with st.sidebar:
        st.markdown("<div style='font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:#00d4aa;margin-bottom:0.75rem;'>DCF Settings</div>", unsafe_allow_html=True)
        dcf_ticker = st.text_input("Ticker","AAPL",key="dcf_t").strip().upper()
        dcf_run    = st.button("▶  Run DCF", type="primary", use_container_width=True, key="dcf_r")
        wacc       = st.slider("WACC (%)",5.0,20.0,10.0,0.5,key="dcf_w")/100
        tgr        = st.slider("Terminal Growth (%)",0.5,5.0,2.5,0.25,key="dcf_tg")/100
        proj_yrs   = st.slider("Projection Years",3,10,5,1,key="dcf_y")
        mos        = st.slider("Margin of Safety (%)",0,40,20,5,key="dcf_m")/100
        override   = st.checkbox("Override growth rate",key="dcf_ov")
        manual_gr  = st.slider("FCF Growth (%)",-10.0,40.0,10.0,0.5,key="dcf_mg")/100 if override else None

    # Hero
    st.markdown("""
    <div class='ib-hero'>
        <div class='ib-hero-eyebrow'>Module 01</div>
        <div class='ib-hero-title'>DCF <em>Valuation</em></div>
        <div class='ib-hero-sub'>Discounted Cash Flow · Intrinsic Value · Sensitivity Analysis</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(TAPE_HTML, unsafe_allow_html=True)

    if dcf_run or "dcf_data" in st.session_state:
        if dcf_run:
            with st.spinner(""):
                st.markdown("<div class='pulse' style='color:#00d4aa;font-size:0.75rem;letter-spacing:0.1em;'>FETCHING DATA…</div>", unsafe_allow_html=True)
                try:
                    d = fetch_dcf(dcf_ticker)
                    st.session_state["dcf_data"] = d
                except Exception as e:
                    st.error(f"❌ {e}"); st.stop()
        d = st.session_state["dcf_data"]
        if not d["fcf_history"]: st.error("No FCF data available."); st.stop()

        bf_auto = d["fcf_history"][-1]
        gr = manual_gr if manual_gr is not None else est_growth(d["fcf_history"])
        bf = bf_auto if bf_auto>0 else np.mean([f for f in d["fcf_history"] if f>0] or [0])
        if bf<=0: st.error("No positive FCF."); st.stop()
        if wacc<=tgr: st.error("WACC must exceed terminal growth."); st.stop()

        res = run_dcf(bf,gr,wacc,tgr,proj_yrs,d["net_debt"],d["shares"])
        intrinsic = res["intrinsic_per_share"]
        mos_price = intrinsic*(1-mos)
        price     = d["current_price"]
        upside    = (intrinsic-price)/price

        section_title(f"{d['name']} <em>·</em> <span style='font-size:1rem;color:#4a5568;'>{d['sector']}</span>")
        if d["description"]:
            with st.expander("Company overview"):
                st.markdown(f"<p style='font-size:0.85rem;color:#8892a4;line-height:1.8;'>{d['description'][:500]}…</p>", unsafe_allow_html=True)

        # Verdict
        if upside>0.25:
            st.markdown(f"<div class='ib-verdict under'>✔  Potentially Undervalued — intrinsic value {upside*100:.1f}% above market price</div>", unsafe_allow_html=True)
        elif upside<-0.25:
            st.markdown(f"<div class='ib-verdict over'>✘  Potentially Overvalued — intrinsic value {abs(upside)*100:.1f}% below market price</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='ib-verdict fair'>~  Fairly Valued — within 25% of market price ({upside*100:+.1f}%)</div>", unsafe_allow_html=True)

        # Metric cards
        uc = "teal" if upside>0 else "red"
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: metric_card("Current Price", fmt_p(price))
        with c2: metric_card("Intrinsic Value", fmt_p(intrinsic), "teal")
        with c3: metric_card(f"MoS Price ({mos*100:.0f}%)", fmt_p(mos_price))
        with c4: metric_card("Upside / Downside", f"{upside*100:+.1f}%", uc)
        with c5: metric_card("Market Cap", fmt_b(d["market_cap"]))

        d1,d2,d3,d4 = st.columns(4)
        with d1: metric_card("P/E Trailing", f"{d['pe_ratio']:.1f}×" if d["pe_ratio"] else "N/A")
        with d2: metric_card("EV/EBITDA", f"{d['ev_ebitda']:.1f}×" if d["ev_ebitda"] else "N/A")
        with d3: metric_card("Beta", f"{d['beta']:.2f}" if d["beta"] else "N/A")
        with d4: metric_card("Analyst Target", fmt_p(d["analyst_target"]), "amber")

        # Charts
        section_title("Cash Flow <em>Analysis</em>")
        ch1,ch2 = st.columns(2)
        with ch1:
            cols = ["#00d4aa" if f>0 else "#ff6b6b" for f in d["fcf_history"]]
            fig = go.Figure(go.Bar(x=d["fcf_years"],y=[f/1e9 for f in d["fcf_history"]],
                marker_color=cols, text=[f"${f/1e9:.1f}B" for f in d["fcf_history"]],
                textposition="outside", textfont=dict(color="#8892a4",size=10)))
            fig.update_layout(**CHART_THEME, title="Historical FCF",
                              title_font=dict(size=13,color="#e8eaf0"))
            st.plotly_chart(fig, use_container_width=True)
        with ch2:
            yl = [f"Y+{y}" for y in res["years"]]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="Projected",x=yl,y=[f/1e9 for f in res["proj_fcf"]],marker_color="#2563eb",opacity=0.7))
            fig2.add_trace(go.Bar(name="Discounted",x=yl,y=[f/1e9 for f in res["disc_fcf"]],marker_color="#00d4aa"))
            fig2.update_layout(**CHART_THEME, barmode="group", title="Projected vs Discounted",
                               title_font=dict(size=13,color="#e8eaf0"),
                               legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#8892a4")))
            st.plotly_chart(fig2, use_container_width=True)

        section_title("Sensitivity <em>Matrix</em>")
        wr=[max(wacc-0.02,0.01),max(wacc-0.01,0.01),wacc,wacc+0.01,wacc+0.02]
        tr=[t for t in [0.010,0.015,0.020,0.025,0.030,0.035,0.040] if t<wacc]
        mat=np.array([[run_dcf(bf,gr,w,t,proj_yrs,d["net_debt"],d["shares"])["intrinsic_per_share"]
                       if w>t else np.nan for w in wr] for t in tr])
        fig3 = go.Figure(go.Heatmap(
            z=mat, x=[f"{w*100:.1f}%" for w in wr], y=[f"{t*100:.1f}%" for t in tr],
            colorscale=[[0,"#ff6b6b"],[0.5,"#f6ad55"],[1,"#00d4aa"]], zmid=price,
            text=[[f"${v:.2f}" if not np.isnan(v) else "—" for v in row] for row in mat],
            texttemplate="%{text}", textfont=dict(size=11),
            colorbar=dict(title="$/share",tickfont=dict(color="#8892a4",size=10))))
        fig3.update_layout(**CHART_THEME, title="WACC × Terminal Growth Rate",
                           title_font=dict(size=13,color="#e8eaf0"),
                           xaxis_title="WACC →", yaxis_title="Terminal Growth →")
        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("📋 Full DCF breakdown"):
            cf_df = pd.DataFrame({"Year":[f"Y+{y}" for y in res["years"]],
                "Projected FCF":[fmt_m(f) for f in res["proj_fcf"]],
                "Discounted FCF":[fmt_m(f) for f in res["disc_fcf"]]})
            st.dataframe(cf_df, use_container_width=True, hide_index=True)
            s = pd.DataFrame({"Component":["PV of FCFs","Terminal Value (PV)","Enterprise Value","Net Debt","Equity Value","Intrinsic / Share"],
                "Value":[fmt_m(res["pv_fcf"]),fmt_m(res["disc_terminal"]),fmt_m(res["enterprise_value"]),fmt_m(d["net_debt"]),fmt_m(res["equity_value"]),fmt_p(intrinsic)]})
            st.dataframe(s, use_container_width=True, hide_index=True)

        st.markdown("<div class='ib-disclaimer'>⚠ For educational and research purposes only. Not financial advice. Always conduct your own due diligence.</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='ib-landing'>
            <div class='ib-landing-title'>Discounted<br><em>Cash Flow</em></div>
            <div class='ib-landing-sub'>Enter a ticker · Adjust assumptions · Run valuation</div>
            <div class='ib-module-pills'>
                <div class='ib-pill'>Intrinsic Value</div>
                <div class='ib-pill'>Sensitivity Analysis</div>
                <div class='ib-pill'>Terminal Value</div>
                <div class='ib-pill'>Margin of Safety</div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — COMPARABLE COMPANIES
# ══════════════════════════════════════════════════════════════
elif page == "🏢  Comparable Companies":

    SECTORS = {
        "Technology (US)":  ["AAPL","MSFT","GOOGL","META","NVDA","ORCL","CRM","ADBE","INTC","AMD","QCOM","TXN","AVGO","NOW","SNOW"],
        "Banking (US)":     ["JPM","BAC","WFC","GS","MS","C","USB","PNC","TFC","COF","SCHW","BK","STT","AXP","DFS"],
        "Healthcare (US)":  ["JNJ","UNH","PFE","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN","GILD","ISRG","MDT","BSX","SYK"],
        "Consumer Retail":  ["AMZN","WMT","TGT","COST","HD","LOW","NKE","SBUX","MCD","YUM","LULU","ROST","TJX","DG","DLTR"],
        "Energy (US)":      ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","PXD","HAL","BKR","DVN","HES"],
        "UK Large Cap":     ["SHEL.L","BP.L","HSBA.L","ULVR.L","AZN.L","GSK.L","DGE.L","RIO.L","BHP.L","VOD.L","LLOY.L","BARC.L","NWG.L"],
    }

    with st.sidebar:
        st.markdown("<div style='font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:#00d4aa;margin-bottom:0.75rem;'>Comps Settings</div>", unsafe_allow_html=True)
        sector = st.selectbox("Sector", list(SECTORS.keys()), key="comp_s")
        custom = st.text_input("Or enter tickers", key="comp_c", placeholder="AAPL, MSFT, GOOGL")
        cr     = st.button("▶  Run Comps", type="primary", use_container_width=True, key="comp_r")

    st.markdown("""
    <div class='ib-hero'>
        <div class='ib-hero-eyebrow'>Module 02</div>
        <div class='ib-hero-title'>Comparable <em>Companies</em></div>
        <div class='ib-hero-sub'>EV/EBITDA · P/E · Debt/EBITDA · FCF Yield · Revenue Multiples</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(TAPE_HTML, unsafe_allow_html=True)

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_comps(tickers):
        rows=[]
        for t in tickers:
            try:
                info=yf.Ticker(t).info
                mc=safe_float(info.get("marketCap")); ev=safe_float(info.get("enterpriseValue"))
                eb=safe_float(info.get("ebitda")); rv=safe_float(info.get("totalRevenue"))
                pe=safe_float(info.get("trailingPE")); fp=safe_float(info.get("forwardPE"))
                db=safe_float(info.get("totalDebt")); fc=safe_float(info.get("freeCashflow"))
                pr=safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
                rows.append({"Ticker":t,"Company":info.get("longName",t)[:28],"Price":pr,
                             "Mkt Cap":mc/1e9 if mc else np.nan,
                             "EV/EBITDA":ev/eb if ev and eb and eb>0 else np.nan,
                             "EV/Revenue":ev/rv if ev and rv and rv>0 else np.nan,
                             "P/E":pe,"P/E Fwd":fp,
                             "Debt/EBITDA":db/eb if db and eb and eb>0 else np.nan,
                             "FCF Yield":fc/mc if fc and mc and mc>0 else np.nan})
            except: continue
        return pd.DataFrame(rows)

    if cr or "comp_df" in st.session_state:
        if cr:
            tks = [x.strip().upper() for x in custom.split(",")] if custom.strip() else SECTORS[sector]
            with st.spinner(""):
                st.markdown("<div class='pulse' style='color:#00d4aa;font-size:0.75rem;letter-spacing:0.1em;'>FETCHING COMPS…</div>", unsafe_allow_html=True)
                df = fetch_comps(tuple(tks))
                st.session_state["comp_df"] = df
                st.session_state["comp_sn"] = sector if not custom.strip() else "Custom"
        df = st.session_state["comp_df"]
        sn = st.session_state.get("comp_sn","")

        section_title(f"Comps Table <em>·</em> <span style='font-size:1rem;color:#4a5568;'>{sn}</span>")
        disp = df.copy()
        def sfmt(x, fmt):
            try:
                return fmt(float(x)) if x is not None and not np.isnan(float(x)) else "N/A"
            except: return "N/A"
        if "Price"    in disp.columns: disp["Price"]    = disp["Price"].apply(lambda x: sfmt(x, lambda v: f"${v:,.2f}"))
        if "Mkt Cap"  in disp.columns: disp["Mkt Cap"]  = disp["Mkt Cap"].apply(lambda x: sfmt(x, lambda v: f"${v:,.1f}B"))
        for c in ["EV/EBITDA","EV/Revenue","P/E","P/E Fwd","Debt/EBITDA"]:
            if c in disp.columns: disp[c] = disp[c].apply(lambda x: sfmt(x, lambda v: f"{v:.1f}×"))
        if "FCF Yield" in disp.columns: disp["FCF Yield"] = disp["FCF Yield"].apply(lambda x: sfmt(x, lambda v: f"{v*100:.1f}%"))
        st.dataframe(disp, use_container_width=True, hide_index=True)

        nc=[c for c in ["EV/EBITDA","EV/Revenue","P/E","P/E Fwd","Debt/EBITDA","FCF Yield"] if c in df.columns]
        def fmt_bench(c, v):
            try:
                if np.isnan(float(v)): return "N/A"
                return f"{float(v)*100:.1f}%" if c=="FCF Yield" else f"{float(v):.1f}x"
            except: return "N/A"
        med={c:df[c].median() for c in nc}
        mn={c:df[c].mean() for c in nc}
        section_title("Sector <em>Benchmarks</em>")
        bd=pd.DataFrame({"Metric":nc,
            "Median":[fmt_bench(c,med[c]) for c in nc],
            "Mean":  [fmt_bench(c,mn[c])  for c in nc]})
        st.dataframe(bd, use_container_width=True, hide_index=True)

        section_title("Visual <em>Comparison</em>")
        c1,c2=st.columns(2)
        with c1:
            nd=df.dropna(subset=["EV/EBITDA"])
            fig=go.Figure(go.Bar(x=nd["Ticker"],y=nd["EV/EBITDA"],
                marker_color=["#00d4aa" if v<med["EV/EBITDA"] else "#2563eb" for v in nd["EV/EBITDA"]],
                text=[f"{v:.1f}×" for v in nd["EV/EBITDA"]],textposition="outside",textfont=dict(size=10,color="#8892a4")))
            fig.add_hline(y=med["EV/EBITDA"],line_dash="dash",line_color="#f6ad55",
                          annotation_text=f"Median {med['EV/EBITDA']:.1f}×",annotation_font_color="#f6ad55")
            fig.update_layout(**CHART_THEME,title="EV/EBITDA",title_font=dict(size=13,color="#e8eaf0"))
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            pd2=df.dropna(subset=["P/E"]); pd2=pd2[pd2["P/E"]<100]
            fig2=go.Figure(go.Bar(x=pd2["Ticker"],y=pd2["P/E"],
                marker_color=["#00d4aa" if v<med["P/E"] else "#ff6b6b" for v in pd2["P/E"]],
                text=[f"{v:.1f}×" for v in pd2["P/E"]],textposition="outside",textfont=dict(size=10,color="#8892a4")))
            fig2.add_hline(y=med["P/E"],line_dash="dash",line_color="#f6ad55",
                           annotation_text=f"Median {med['P/E']:.1f}×",annotation_font_color="#f6ad55")
            fig2.update_layout(**CHART_THEME,title="P/E Ratio",title_font=dict(size=13,color="#e8eaf0"))
            st.plotly_chart(fig2,use_container_width=True)

        c3,c4=st.columns(2)
        with c3:
            dd=df.dropna(subset=["Debt/EBITDA"])
            fig3=go.Figure(go.Bar(x=dd["Ticker"],y=dd["Debt/EBITDA"],
                marker_color=["#ff6b6b" if v>3.0 else "#00d4aa" for v in dd["Debt/EBITDA"]],
                text=[f"{v:.1f}×" for v in dd["Debt/EBITDA"]],textposition="outside",textfont=dict(size=10,color="#8892a4")))
            fig3.add_hline(y=3.0,line_dash="dash",line_color="#f6ad55",
                           annotation_text="3× threshold",annotation_font_color="#f6ad55")
            fig3.update_layout(**CHART_THEME,title="Debt/EBITDA",title_font=dict(size=13,color="#e8eaf0"))
            st.plotly_chart(fig3,use_container_width=True)
        with c4:
            fd=df.dropna(subset=["FCF Yield"])
            fig4=go.Figure(go.Bar(x=fd["Ticker"],y=fd["FCF Yield"]*100,
                marker_color=["#00d4aa" if v>5.0 else "#2563eb" for v in fd["FCF Yield"]*100],
                text=[f"{v:.1f}%" for v in fd["FCF Yield"]*100],textposition="outside",textfont=dict(size=10,color="#8892a4")))
            fig4.add_hline(y=5.0,line_dash="dash",line_color="#f6ad55",
                           annotation_text="5% threshold",annotation_font_color="#f6ad55")
            fig4.update_layout(**CHART_THEME,title="FCF Yield",title_font=dict(size=13,color="#e8eaf0"))
            st.plotly_chart(fig4,use_container_width=True)

        st.markdown("<div class='ib-disclaimer'>⚠ For educational purposes only. Not financial advice.</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='ib-landing'>
            <div class='ib-landing-title'>Comparable<br><em>Companies</em></div>
            <div class='ib-landing-sub'>Select a sector or enter custom tickers</div>
            <div class='ib-module-pills'>
                <div class='ib-pill'>EV/EBITDA</div><div class='ib-pill'>P/E Ratio</div>
                <div class='ib-pill'>Debt/EBITDA</div><div class='ib-pill'>FCF Yield</div>
            </div>
            <div class='ib-stats-row'>
                <div class='ib-stat'><span class='ib-stat-num'>6</span><span class='ib-stat-lbl'>Sectors</span></div>
                <div class='ib-stat'><span class='ib-stat-num'>15</span><span class='ib-stat-lbl'>Peers / Sector</span></div>
                <div class='ib-stat'><span class='ib-stat-num'>6</span><span class='ib-stat-lbl'>Multiples</span></div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — 3-STATEMENT MODEL
# ══════════════════════════════════════════════════════════════
elif page == "📑  3-Statement Model":

    with st.sidebar:
        st.markdown("<div style='font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:#00d4aa;margin-bottom:0.75rem;'>Model Settings</div>", unsafe_allow_html=True)
        stmt_t = st.text_input("Ticker","AAPL",key="stmt_t").strip().upper()
        stmt_r = st.button("▶  Load Statements", type="primary", use_container_width=True, key="stmt_r")
        st.markdown("<div style='font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:#00d4aa;margin:1rem 0 0.75rem;'>Forecast Assumptions</div>", unsafe_allow_html=True)
        rev_gr    = st.slider("Revenue Growth (%)",-10.0,40.0,8.0,0.5,key="rg")/100
        op_margin = st.slider("Operating Margin (%)",1.0,50.0,20.0,0.5,key="om")/100
        tax_rate  = st.slider("Tax Rate (%)",5.0,35.0,21.0,0.5,key="tr")/100
        capex_pct = st.slider("Capex % Revenue",1.0,20.0,5.0,0.5,key="cp")/100
        da_pct    = st.slider("D&A % Revenue",1.0,15.0,4.0,0.5,key="da")/100
        nwc_pct   = st.slider("NWC Change %",-5.0,5.0,1.0,0.5,key="nw")/100

    st.markdown("""
    <div class='ib-hero'>
        <div class='ib-hero-eyebrow'>Module 03</div>
        <div class='ib-hero-title'>3-Statement <em>Model</em></div>
        <div class='ib-hero-sub'>Income Statement · Balance Sheet · Cash Flow · 5-Year Forecast</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(TAPE_HTML, unsafe_allow_html=True)

    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_stmts(ticker):
        tk=yf.Ticker(ticker)
        return tk.info, tk.income_stmt, tk.balance_sheet, tk.cashflow

    if stmt_r or "stmt_data" in st.session_state:
        if stmt_r:
            with st.spinner(""):
                st.markdown("<div class='pulse' style='color:#00d4aa;font-size:0.75rem;letter-spacing:0.1em;'>LOADING STATEMENTS…</div>", unsafe_allow_html=True)
                try:
                    info,inc,bal,cf=fetch_stmts(stmt_t)
                    st.session_state["stmt_data"]=(info,inc,bal,cf,stmt_t)
                except Exception as e:
                    st.error(f"❌ {e}"); st.stop()
        info,inc,bal,cf,lt=st.session_state["stmt_data"]
        section_title(f"{info.get('longName',lt)} <em>·</em> <span style='font-size:1rem;color:#4a5568;'>Historical Statements</span>")

        for label, df_raw, rows in [
            ("Income Statement (USD Billions)", inc, ["Total Revenue","Gross Profit","Operating Income","Net Income","EBITDA"]),
            ("Cash Flow Statement (USD Billions)", cf, ["Operating Cash Flow","Capital Expenditure","Total Cash From Operating Activities","Capital Expenditures"]),
            ("Balance Sheet (USD Billions)", bal, ["Total Assets","Total Liabilities Net Minority Interest","Stockholders Equity","Total Debt","Cash And Cash Equivalents"]),
        ]:
            try:
                d2=df_raw.copy(); d2.columns=[str(c.year) for c in d2.columns]
                d2=(d2/1e9).round(2)
                filt=d2.loc[[r for r in rows if r in d2.index]]
                st.markdown(f"<div style='font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;color:#00d4aa;margin:1.5rem 0 0.5rem;'>{label}</div>", unsafe_allow_html=True)
                st.dataframe(filt, use_container_width=True)
            except: pass

        section_title("5-Year <em>Forecast</em>")
        try:
            br=None
            for rn in ["Total Revenue","Revenue"]:
                try: br=float(inc.loc[rn].iloc[0]); break
                except: continue
            if br is None or np.isnan(br): raise ValueError
        except:
            st.error("Could not extract base revenue."); st.stop()

        yrs=["Y+1","Y+2","Y+3","Y+4","Y+5"]
        revs,ops,ebs,nets,fcfs=[],[],[],[],[]
        r=br
        for _ in range(5):
            r*=(1+rev_gr); oi=r*op_margin; eb=oi+r*da_pct
            ni=oi*(1-tax_rate); fc=ni+r*da_pct-r*capex_pct-r*nwc_pct
            revs.append(r/1e9); ops.append(oi/1e9); ebs.append(eb/1e9)
            nets.append(ni/1e9); fcfs.append(fc/1e9)

        fd=pd.DataFrame({"Year":yrs,
            "Revenue ($B)":[f"${v:.2f}B" for v in revs],
            "Op. Income ($B)":[f"${v:.2f}B" for v in ops],
            "EBITDA ($B)":[f"${v:.2f}B" for v in ebs],
            "Net Income ($B)":[f"${v:.2f}B" for v in nets],
            "FCF ($B)":[f"${v:.2f}B" for v in fcfs]})
        st.dataframe(fd, use_container_width=True, hide_index=True)

        fig=go.Figure()
        fig.add_trace(go.Bar(name="Revenue",x=yrs,y=revs,marker_color="#2563eb",opacity=0.6))
        fig.add_trace(go.Bar(name="EBITDA",x=yrs,y=ebs,marker_color="#00d4aa"))
        fig.add_trace(go.Bar(name="FCF",x=yrs,y=fcfs,marker_color="#f6ad55"))
        fig.update_layout(**CHART_THEME,barmode="group",title="5-Year Forecast",
                          title_font=dict(size=13,color="#e8eaf0"),
                          legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#8892a4")))
        st.plotly_chart(fig,use_container_width=True)

        section_title("Sensitivity <em>Heatmap</em>")
        st.caption("Year 5 FCF across different Revenue Growth × Operating Margin combinations")
        rr=[r/100 for r in np.arange(max(rev_gr*100-6,1),rev_gr*100+8,2)]
        mr=[m/100 for m in np.arange(max(op_margin*100-8,1),op_margin*100+10,2)]
        sens=[]
        for mg in mr:
            row=[]
            for rg in rr:
                rv=br
                for _ in range(5): rv*=(1+rg)
                row.append((rv*mg*(1-tax_rate)+rv*da_pct-rv*capex_pct-rv*nwc_pct)/1e9)
            sens.append(row)
        sens=np.array(sens)
        fig_s=go.Figure(go.Heatmap(
            z=sens, x=[f"{r*100:.1f}%" for r in rr], y=[f"{m*100:.1f}%" for m in mr],
            colorscale=[[0,"#ff6b6b"],[0.5,"#f6ad55"],[1,"#00d4aa"]],
            text=[[f"${v:.1f}B" for v in row] for row in sens],
            texttemplate="%{text}", textfont=dict(size=11),
            colorbar=dict(title="FCF $B",tickfont=dict(color="#8892a4",size=10))))
        fig_s.update_layout(**CHART_THEME,title="Year 5 FCF — Revenue Growth × Operating Margin",
                            title_font=dict(size=13,color="#e8eaf0"),
                            xaxis_title="Revenue Growth →",yaxis_title="Operating Margin →")
        st.plotly_chart(fig_s,use_container_width=True)
        st.markdown("<div class='ib-disclaimer'>⚠ For educational purposes only. Not financial advice.</div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='ib-landing'>
            <div class='ib-landing-title'>3-Statement<br><em>Model</em></div>
            <div class='ib-landing-sub'>Load historical financials and forecast forward</div>
            <div class='ib-module-pills'>
                <div class='ib-pill'>Income Statement</div><div class='ib-pill'>Balance Sheet</div>
                <div class='ib-pill'>Cash Flow</div><div class='ib-pill'>Sensitivity Heatmap</div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — EARNINGS SCREENER
# ══════════════════════════════════════════════════════════════
elif page == "🔍  Earnings Screener":

    SP500 = [
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
        st.markdown("<div style='font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:#00d4aa;margin-bottom:0.75rem;'>Screener Filters</div>", unsafe_allow_html=True)
        max_de   = st.slider("Max Debt/EBITDA",0.5,8.0,3.0,0.5,key="sd")
        min_fy   = st.slider("Min FCF Yield (%)",0.0,15.0,5.0,0.5,key="sf")/100
        max_pe   = st.slider("Max P/E",5.0,60.0,25.0,1.0,key="sp")
        min_cap  = st.slider("Min Market Cap ($B)",1.0,100.0,10.0,1.0,key="sc")
        scan_n   = st.slider("Companies to scan",20,90,50,10,key="sn")
        scr_run  = st.button("▶  Run Screener", type="primary", use_container_width=True, key="sr")

    st.markdown("""
    <div class='ib-hero'>
        <div class='ib-hero-eyebrow'>Module 04</div>
        <div class='ib-hero-title'>Earnings <em>Screener</em></div>
        <div class='ib-hero-sub'>Value Criteria · FCF Yield · Debt/EBITDA · S&P 500 Universe</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(TAPE_HTML, unsafe_allow_html=True)

    if scr_run:
        tks=SP500[:scan_n]
        prog=st.progress(0,text="Initialising scan…")
        results=[]
        for i,t in enumerate(tks):
            prog.progress((i+1)/len(tks), text=f"Scanning {t}…")
            try:
                info=yf.Ticker(t).info
                mc=safe_float(info.get("marketCap"))
                if np.isnan(mc) or mc<min_cap*1e9: continue
                eb=safe_float(info.get("ebitda")); db=safe_float(info.get("totalDebt"))
                fc=safe_float(info.get("freeCashflow")); pe=safe_float(info.get("trailingPE"))
                ev=safe_float(info.get("enterpriseValue"))
                pr=safe_float(info.get("regularMarketPrice") or info.get("currentPrice"))
                de=db/eb if db and eb and eb>0 else np.nan
                fy=fc/mc if fc and mc and mc>0 else np.nan
                ee=ev/eb if ev and eb and eb>0 else np.nan
                ok=(not np.isnan(fy) and not np.isnan(de) and
                    de<=max_de and fy>=min_fy and (np.isnan(pe) or pe<=max_pe))
                if ok:
                    results.append({"Ticker":t,"Company":info.get("longName",t)[:30],
                                    "Sector":info.get("sector","N/A"),"Price":pr,
                                    "Mkt Cap ($B)":mc/1e9,"P/E":pe,
                                    "EV/EBITDA":ee,"Debt/EBITDA":de,"FCF Yield":fy})
            except: continue
        prog.empty()
        st.session_state["scr_res"]=pd.DataFrame(results)
        st.session_state["scr_p"]=(max_de,min_fy,max_pe,min_cap,scan_n)

    if "scr_res" in st.session_state:
        df=st.session_state["scr_res"]
        p=st.session_state.get("scr_p",(max_de,min_fy,max_pe,min_cap,scan_n))

        c1,c2,c3,c4=st.columns(4)
        with c1: metric_card("Passed", str(len(df)), "teal")
        with c2: metric_card("Max Debt/EBITDA", f"{p[0]:.1f}×")
        with c3: metric_card("Min FCF Yield", f"{p[1]*100:.1f}%")
        with c4: metric_card("Max P/E", f"{p[2]:.0f}×")

        if df.empty:
            st.warning("No companies passed. Try relaxing the filters.")
        else:
            section_title(f"<em>{len(df)}</em> Companies Passed")
            disp=df.copy().sort_values("FCF Yield",ascending=False)
            disp["Price"]       =disp["Price"].apply(lambda x: f"${x:,.2f}" if not np.isnan(x) else "N/A")
            disp["Mkt Cap ($B)"]=disp["Mkt Cap ($B)"].apply(lambda x: f"${x:,.1f}B" if not np.isnan(x) else "N/A")
            disp["P/E"]         =disp["P/E"].apply(lambda x: f"{x:.1f}×" if not np.isnan(x) else "N/A")
            disp["EV/EBITDA"]   =disp["EV/EBITDA"].apply(lambda x: f"{x:.1f}×" if not np.isnan(x) else "N/A")
            disp["Debt/EBITDA"] =disp["Debt/EBITDA"].apply(lambda x: f"{x:.1f}×" if not np.isnan(x) else "N/A")
            disp["FCF Yield"]   =disp["FCF Yield"].apply(lambda x: f"{x*100:.1f}%" if not np.isnan(x) else "N/A")
            st.dataframe(disp, use_container_width=True, hide_index=True)

            section_title("Visual <em>Analysis</em>")
            pl=df.sort_values("FCF Yield",ascending=False).head(20)
            a1,a2=st.columns(2)
            with a1:
                fig=go.Figure(go.Bar(x=pl["Ticker"],y=pl["FCF Yield"]*100,
                    marker_color="#00d4aa",
                    text=[f"{v:.1f}%" for v in pl["FCF Yield"]*100],
                    textposition="outside",textfont=dict(size=10,color="#8892a4")))
                fig.add_hline(y=p[1]*100,line_dash="dash",line_color="#f6ad55",
                              annotation_text=f"Min {p[1]*100:.1f}%",annotation_font_color="#f6ad55")
                fig.update_layout(**CHART_THEME,title="FCF Yield",title_font=dict(size=13,color="#e8eaf0"))
                st.plotly_chart(fig,use_container_width=True)
            with a2:
                dd=df.dropna(subset=["Debt/EBITDA"]).sort_values("Debt/EBITDA")
                fig2=go.Figure(go.Bar(x=dd["Ticker"],y=dd["Debt/EBITDA"],
                    marker_color=["#00d4aa" if v<2.0 else "#f6ad55" for v in dd["Debt/EBITDA"]],
                    text=[f"{v:.1f}×" for v in dd["Debt/EBITDA"]],
                    textposition="outside",textfont=dict(size=10,color="#8892a4")))
                fig2.add_hline(y=p[0],line_dash="dash",line_color="#ff6b6b",
                               annotation_text=f"Max {p[0]:.1f}×",annotation_font_color="#ff6b6b")
                fig2.update_layout(**CHART_THEME,title="Debt/EBITDA",title_font=dict(size=13,color="#e8eaf0"))
                st.plotly_chart(fig2,use_container_width=True)

            section_title("Sector <em>Breakdown</em>")
            sc=df["Sector"].value_counts().reset_index(); sc.columns=["Sector","Count"]
            fig3=go.Figure(go.Pie(labels=sc["Sector"],values=sc["Count"],
                marker=dict(colors=["#00d4aa","#2563eb","#f6ad55","#ff6b6b","#8892a4",
                                    "#a78bfa","#34d399","#fb923c","#60a5fa","#f472b6"]),
                textfont=dict(color="#e8eaf0",size=12),hole=0.4))
            fig3.update_layout(**CHART_THEME,title="Companies by Sector",
                               title_font=dict(size=13,color="#e8eaf0"))
            st.plotly_chart(fig3,use_container_width=True)
            st.info("💡 Take any ticker from the results and run a full DCF on the **📊 DCF Valuation** page.")

        st.markdown("<div class='ib-disclaimer'>⚠ For educational purposes only. Not financial advice.</div>", unsafe_allow_html=True)
    elif not scr_run:
        st.markdown("""
        <div class='ib-landing'>
            <div class='ib-landing-title'>Earnings<br><em>Screener</em></div>
            <div class='ib-landing-sub'>Filter the S&P 500 by value criteria</div>
            <div class='ib-module-pills'>
                <div class='ib-pill'>Debt/EBITDA</div><div class='ib-pill'>FCF Yield</div>
                <div class='ib-pill'>P/E Filter</div><div class='ib-pill'>Sector Breakdown</div>
            </div>
            <div class='ib-stats-row'>
                <div class='ib-stat'><span class='ib-stat-num'>90+</span><span class='ib-stat-lbl'>S&P 500 Names</span></div>
                <div class='ib-stat'><span class='ib-stat-num'>4</span><span class='ib-stat-lbl'>Filters</span></div>
                <div class='ib-stat'><span class='ib-stat-num'>Live</span><span class='ib-stat-lbl'>Data</span></div>
            </div>
        </div>""", unsafe_allow_html=True)
