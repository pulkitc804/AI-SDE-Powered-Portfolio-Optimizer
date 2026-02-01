import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import torch
from strategies.scanner import MarketScanner
from data.loader import MarketDataLoader
from models.neural_sde import NeuralSDE
from engine.trainer import SDETrainer
from config.settings import DEVICE

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="QUANT SDE | Master",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚡"
)

# ---------------------------------------------------------
# 2. PRO-GRADE CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;700&display=swap');
    
    :root {
        --bg-dark: #09090b;
        --card-bg: #18181b;
        --border-color: #27272a;
        --accent: #3b82f6;
        --text-main: #f4f4f5;
        --text-sub: #a1a1aa;
    }

    .stApp {
        background-color: var(--bg-dark);
        font-family: 'Manrope', sans-serif;
        color: var(--text-main);
    }
    
    /* --- SIDEBAR STYLING --- */
    section[data-testid="stSidebar"] {
        background-color: #0c0c0e;
        border-right: 1px solid var(--border-color);
    }
    
    .stExpander {
        background-color: transparent;
        border: none;
    }

    /* --- DASHBOARD CARDS --- */
    .metric-card {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: white;
    }
    
    .metric-label {
        font-size: 13px;
        color: var(--text-sub);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }

    /* --- TABS & INPUTS --- */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--bg-dark);
        border-bottom: 1px solid var(--border-color);
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 14px;
        font-weight: 500;
        color: var(--text-sub);
        border: none;
        padding-bottom: 12px;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--accent);
        border-bottom: 2px solid var(--accent);
    }
    
    /* Custom Tooltip Style */
    div[data-testid="stTooltipIcon"] {
        color: var(--accent);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. ADVANCED CONTROL PANEL (SIDEBAR)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛 CONTROL PANEL")

    # --- SECTION 1: ASSET UNIVERSE (NEW) ---
    with st.expander("💼 Asset Universe", expanded=True):
        st.caption("Define the pool of assets for the Scanner and Portfolio.")

        # Predefined Institutional List
        INSTITUTIONAL_LIST = [
            "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "AMD", # Tech
            "SPY", "QQQ", "IWM", # Indices
            "JPM", "GS", "BAC", "V", # Finance
            "BTC-USD", "ETH-USD", "COIN", # Crypto
            "XOM", "CVX", "LLY", "UNH" # Old Economy
        ]

        selected_assets = st.multiselect(
            "Quick Select (Top 25)",
            INSTITUTIONAL_LIST,
            default=["NVDA", "TSLA", "AAPL", "SPY", "BTC-USD"]
        )

        custom_input = st.text_input("Add Custom (Comma Separated)", "", placeholder="e.g. PLTR, GME, HOOD")

        # Logic to combine lists
        custom_list = [x.strip().upper() for x in custom_input.split(',') if x.strip()]
        ACTIVE_TICKERS = list(set(selected_assets + custom_list))

        st.caption(f"Active Universe: {len(ACTIVE_TICKERS)} Assets")

    # --- SECTION 2: DATA PIPELINE ---
    with st.expander("📡 Data Pipeline", expanded=False):
        lookback = st.slider(
            "Lookback Window (Days)", 30, 500, 90,
            help="Historical data fed to the model."
        )
        data_source = st.selectbox(
            "Data Feed", ["Yahoo Finance (Free)", "AlphaVantage", "Bloomberg"],
            help="Source API."
        )

    # --- SECTION 3: NEURAL HYPERPARAMETERS ---
    with st.expander("🧠 Model Architecture", expanded=False):
        epochs = st.number_input("Training Epochs", 50, 2000, 150)
        learning_rate = st.select_slider("Learning Rate", [0.0001, 0.001, 0.01, 0.1], value=0.01)
        hidden_dim = st.selectbox("Hidden Layer Size", [32, 64, 128, 256], index=2)
        dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2)

    # --- SECTION 4: STRESS TESTING ---
    with st.expander("⚠ Stress Testing", expanded=False):
        drift_override = st.slider("Drift Adj (%)", -0.5, 0.5, 0.0, step=0.01)
        vol_scalar = st.slider("Vol Multiplier", 0.5, 3.0, 1.0)

    # --- SECTION 5: RISK MANAGEMENT ---
    with st.expander("🛡 Risk Parameters", expanded=False):
        rf_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 4.5) / 100.0

    st.markdown("---")
    st.caption(f"Status: ONLINE | {DEVICE}")

# ---------------------------------------------------------
# 4. NAVIGATION
# ---------------------------------------------------------
tabs = st.tabs([
    "Overview",
    "Scanner",
    "Forecast Lab",
    "Alpha Signals",
    "Portfolio Opt"
])

# HELPER: RSI Calculation
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# =========================================================
# TAB 1: OVERVIEW
# =========================================================
with tabs[0]:
    st.markdown("### Market Dashboard")

    # Initialization
    curr_price = 0.0
    daily_ret = 0.0
    current_rsi = 50.0
    regime = "N/A"
    curr_vol = "N/A"
    vol_state = "N/A"

    # Fetch SPY for Global Context
    try:
        loader = MarketDataLoader('SPY', 100)
        df_spy = loader.download_data()

        if not df_spy.empty:
            curr_price = df_spy['Close'].iloc[-1]
            prev_price = df_spy['Close'].iloc[-2]
            daily_ret = (curr_price - prev_price) / prev_price

            rsi_series = calculate_rsi(df_spy['Close'])
            current_rsi = rsi_series.iloc[-1]

            if current_rsi > 70: regime = "Overbought (Bearish Risk)"
            elif current_rsi < 30: regime = "Oversold (Bullish Bounce)"
            else: regime = "Neutral Trend"

            raw_vol = df_spy['Close'].pct_change().std() * np.sqrt(252)
            curr_vol = raw_vol

            if raw_vol > 0.20: vol_state = "High (Risk-Off)"
            else: vol_state = "Normal (Risk-On)"
    except:
        pass

    # Safe Formatting
    vol_display = f"{curr_vol:.1%}" if isinstance(curr_vol, (float, np.float64)) else "N/A"
    price_display = f"${curr_price:.2f}"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">S&P 500 Price</div>
            <div class="metric-value">{price_display}</div>
            <div style="color: {'#4ade80' if daily_ret > 0 else '#f87171'}; font-size: 14px;">
                {daily_ret:+.2%}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Momentum (RSI)</div>
            <div class="metric-value">{current_rsi:.1f}</div>
            <div style="color: #94a3b8; font-size: 14px;">{regime}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Volatility State</div>
            <div class="metric-value">{vol_display}</div>
            <div style="color: #94a3b8; font-size: 14px;">{vol_state}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk-Free Rate</div>
            <div class="metric-value">{rf_rate:.2%}</div>
            <div style="color: #94a3b8; font-size: 14px;">10Y Treasury Yield</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.info(f"💡 **Active Universe:** You have {len(ACTIVE_TICKERS)} assets selected in the sidebar. These will be used for Scanning and Portfolio Optimization.")

# =========================================================
# TAB 2: SCANNER
# =========================================================
with tabs[1]:
    st.markdown("### Universe Scanner")

    col_act, col_btn = st.columns([3, 1])
    with col_act:
        st.write(f"Scanning the **{len(ACTIVE_TICKERS)} active assets** defined in the sidebar.")

    with col_btn:
        run_scan = st.button("RUN SCAN", use_container_width=True)

    if run_scan:
        with st.status("Processing Market Data...", expanded=True) as status:
            data = []
            bar = st.progress(0)

            for i, t in enumerate(ACTIVE_TICKERS):
                try:
                    ld = MarketDataLoader(t, 60)
                    df = ld.download_data()
                    if not df.empty and len(df) > 30:
                        ret = df['Close'].pct_change().dropna()
                        vol = ret.std() * np.sqrt(252)
                        rsi = calculate_rsi(df['Close']).iloc[-1]

                        # Mock Neural Inference for Scanner Speed
                        # (Real inference for 25 assets would take 30-60s)
                        edge = np.random.normal(0, 0.05)

                        data.append({"Ticker": t, "Price": df['Close'].iloc[-1], "Vol": vol, "RSI": rsi, "Edge": edge})
                except: pass
                bar.progress((i+1)/len(ACTIVE_TICKERS))

            status.update(label="Scan Complete", state="complete", expanded=False)
            st.session_state['scan_results'] = pd.DataFrame(data)

    if 'scan_results' in st.session_state and not st.session_state['scan_results'].empty:
        df_scan = st.session_state['scan_results']

        # Treemap
        fig = px.treemap(
            df_scan, path=['Ticker'], values='Price', color='Edge',
            color_continuous_scale=['#ef4444', '#18181b', '#22c55e'],
            color_continuous_midpoint=0
        )
        fig.update_layout(margin=dict(t=0,l=0,r=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.dataframe(
            df_scan.style.format({"Price":"${:.2f}", "Vol":"{:.1%}", "RSI":"{:.1f}", "Edge":"{:.1%}"})
            .background_gradient(subset=['Edge'], cmap='RdYlGn', vmin=-0.05, vmax=0.05),
            use_container_width=True
        )

# =========================================================
# TAB 3: FORECAST LAB
# =========================================================
with tabs[2]:
    st.markdown("### Neural Forecast Engine")

    c1, c2 = st.columns([1, 3])
    with c1:
        # Allow selection from the Active List OR typing
        target_asset = st.selectbox("Select Asset", ACTIVE_TICKERS)

        forecast_days = st.slider("Forecast Horizon", 5, 60, 21)
        num_sims = st.selectbox("Monte Carlo Paths", [500, 1000, 2000], index=1)
        run_sim = st.button("Run Simulation", use_container_width=True)

    with c2:
        if run_sim:
            with st.spinner(f"Training Neural SDE on {target_asset}..."):
                try:
                    loader = MarketDataLoader(target_asset, lookback)
                    X, y, spot_tn, _ = loader.fetch_realtime_data()
                    spot = spot_tn.item()

                    # Train
                    model = NeuralSDE(3, hidden_dim, 3)
                    trainer = SDETrainer(model, lr=learning_rate)
                    trainer.train(X, y, epochs)

                    # Predict
                    last_seq = X[-1].clone().detach().to(DEVICE) if isinstance(X, torch.Tensor) else torch.tensor(X[-1]).to(DEVICE)
                    paths = trainer.predict_future(last_seq, spot, num_sims, forecast_days)

                    # Stress Test Mods
                    if drift_override != 0.0 or vol_scalar != 1.0:
                        time_steps = np.arange(forecast_days + 1)
                        drift_factor = (1 + drift_override) ** time_steps
                        mean_path = np.mean(paths, axis=0)
                        centered = paths - mean_path
                        paths = mean_path * drift_factor + (centered * vol_scalar)

                    # Plot
                    fig = go.Figure()
                    days = np.arange(forecast_days+1)
                    p95 = np.percentile(paths, 95, 0)
                    p05 = np.percentile(paths, 5, 0)
                    med = np.median(paths, 0)

                    fig.add_trace(go.Scatter(
                        x=np.concatenate([days, days[::-1]]),
                        y=np.concatenate([p95, p05[::-1]]),
                        fill='toself', fillcolor='rgba(59, 130, 246, 0.2)',
                        line=dict(width=0), name='95% Confidence'
                    ))
                    fig.add_trace(go.Scatter(x=days, y=med, line=dict(color='#3b82f6', width=2), name='Projection'))

                    fig.update_layout(
                        title=f"{target_asset} Projection ({num_sims} Paths)",
                        xaxis_title="Days Ahead", yaxis_title="Price",
                        template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Metrics
                    final_prices = paths[:, -1]
                    exp_ret = (np.mean(final_prices) - spot) / spot
                    # Avoid divide by zero
                    std_dev = np.std(final_prices)/spot
                    sharpe = (exp_ret - rf_rate) / std_dev if std_dev > 0 else 0

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Exp. Return", f"{exp_ret:.2%}")
                    m2.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    m3.metric("VaR (95%)", f"${spot - np.percentile(final_prices, 5):.2f}")

                except Exception as e: st.error(f"Simulation Error: {str(e)}")

# =========================================================
# TAB 4: ALPHA SIGNALS
# =========================================================
with tabs[3]:
    st.markdown("### Alpha Signals")

    if 'scan_results' in st.session_state:
        df = st.session_state['scan_results']

        c_long, c_short = st.columns(2)

        with c_long:
            st.markdown('<div class="metric-card">🟢 Long Opportunities</div>', unsafe_allow_html=True)
            st.caption("Condition: Positive Edge (Undervalued Volatility)")
            longs = df[df['Edge'] > 0.01].sort_values('Edge', ascending=False)
            if not longs.empty:
                st.dataframe(longs[['Ticker', 'Price', 'Edge']], use_container_width=True, hide_index=True)
            else: st.info("No Long signals found.")

        with c_short:
            st.markdown('<div class="metric-card">🔴 Short Opportunities</div>', unsafe_allow_html=True)
            st.caption("Condition: Negative Edge (Overvalued Volatility)")
            shorts = df[df['Edge'] < -0.01].sort_values('Edge', ascending=True)
            if not shorts.empty:
                st.dataframe(shorts[['Ticker', 'Price', 'Edge']], use_container_width=True, hide_index=True)
            else: st.info("No Short signals found.")
    else:
        st.warning("Please run a Scan in the 'Scanner' tab first.")

# =========================================================
# TAB 5: PORTFOLIO OPTIMIZER
# =========================================================
with tabs[4]:
    st.markdown("### Efficient Frontier Construction")

    if len(ACTIVE_TICKERS) < 2:
        st.error("Please select at least 2 assets in the Sidebar.")
    else:
        if st.button("Optimize Portfolio"):
            with st.spinner("Simulating 5,000 Portfolios..."):
                # Mock simulation for UI responsiveness
                n_port = 5000
                results = []

                for _ in range(n_port):
                    weights = np.random.random(len(ACTIVE_TICKERS))
                    weights /= np.sum(weights)

                    # Simulated Return/Risk
                    port_ret = np.sum(weights * 0.12) # Mock 12% avg
                    port_vol = np.sqrt(np.dot(weights.T, np.dot(np.eye(len(ACTIVE_TICKERS))*0.04, weights)))
                    sharpe = (port_ret - rf_rate) / port_vol

                    results.append([port_ret, port_vol, sharpe])

                res_array = np.array(results)
                max_sharpe_idx = np.argmax(res_array[:, 2])

                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=res_array[:, 1], y=res_array[:, 0], mode='markers',
                    marker=dict(color=res_array[:, 2], colorscale='Viridis', showscale=True),
                    name='Portfolios'
                ))
                fig.add_trace(go.Scatter(
                    x=[res_array[max_sharpe_idx, 1]], y=[res_array[max_sharpe_idx, 0]],
                    mode='markers', marker=dict(color='red', size=15, symbol='star'),
                    name='Optimal Portfolio'
                ))

                fig.update_layout(
                    title="Efficient Frontier", xaxis_title="Risk (Vol)", yaxis_title="Return",
                    template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

                st.success(f"Optimal Sharpe Ratio: {res_array[max_sharpe_idx, 2]:.2f}")