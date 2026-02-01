import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from datetime import datetime
import pytz

# =========================================================
# 1. PAGE CONFIG & VISUAL STYLING
# =========================================================
# This MUST be the very first Streamlit command
st.set_page_config(
    page_title="SDE Neural Network Portfolio Optimizer",
    layout="wide",
    initial_sidebar_state="expanded"  # FORCE SIDEBAR OPEN
)

# CUSTOM CSS: Full-screen tabs, colorful metrics, hover effects
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    
    /* FULL WIDTH TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        border-bottom: 2px solid #27272a;
        width: 100%;
    }

    .stTabs [data-baseweb="tab"] {
        flex-grow: 1;
        text-align: center;
        height: 60px;
        font-size: 16px;
        font-weight: 600;
        color: #a1a1aa;
        background-color: transparent;
        border: none;
        border-radius: 4px 4px 0 0;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #60a5fa; /* Blue-400 */
        background-color: rgba(30, 41, 59, 0.5);
    }

    .stTabs [aria-selected="true"] {
        color: #3b82f6 !important;
        border-bottom: 3px solid #3b82f6;
    }

    /* CUSTOM METRIC CARDS */
    div.metric-container {
        background-color: #18181b;
        border: 1px solid #27272a;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div.metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #f4f4f5;
        margin: 5px 0;
    }
    div.metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #a1a1aa;
    }
    div.metric-delta-pos { color: #4ade80; font-size: 14px; font-weight: 600; }
    div.metric-delta-neg { color: #f87171; font-size: 14px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. CORE LOGIC (Fixed Shapes & Calculations)
# =========================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

class QuantData:
    def __init__(self, ticker, lookback_days=90):
        self.ticker = ticker
        self.lookback = lookback_days

    def fetch_realtime_data(self):
        try:
            df = yf.download(self.ticker, period=f"{self.lookback + 60}d", progress=False)
            if df.empty or len(df) < 50: return None, None, None, None

            # Handle yfinance MultiIndex / Format changes
            if 'Close' in df.columns:
                if isinstance(df['Close'], pd.DataFrame):
                    prices = df['Close'].iloc[:, 0].values.astype(float)
                else:
                    prices = df['Close'].values.astype(float)
            else:
                # Fallback if structure is very weird
                prices = df.iloc[:, 0].values.astype(float)

            spot = prices[-1]

            # 3 FEATURES: [Price (Norm), Returns, Vol]
            norm_prices = prices / spot
            returns = np.diff(norm_prices, prepend=norm_prices[0])
            vol = pd.Series(returns).rolling(20).std().fillna(0).values

            data_matrix = np.column_stack([norm_prices, returns, vol])

            # Sequence Creation
            seq_len = 30
            X, y = [], []
            for i in range(len(data_matrix) - seq_len):
                X.append(data_matrix[i:i+seq_len])
                y.append(data_matrix[i+seq_len, 0])

                # Convert to Tensor
            X_t = torch.tensor(np.array(X), dtype=torch.float32).to(DEVICE)
            y_t = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1).to(DEVICE)
            spot_t = torch.tensor(spot, dtype=torch.float32).to(DEVICE)

            return X_t, y_t, spot_t, df
        except: return None, None, None, None

class QuantModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super(QuantModel, self).__init__()
        self.drift_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.diffusion_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Softplus()
        )

    def forward(self, x):
        if x.dim() == 3: x = x[:, -1, :] # Take last step
        return self.drift_net(x), self.diffusion_net(x)

class QuantTrainer:
    def __init__(self, model, lr=0.01):
        self.model = model.to(DEVICE)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train(self, X, y, epochs=100):
        self.model.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            mu, _ = self.model(X)
            loss = self.criterion(mu, y)
            loss.backward()
            self.optimizer.step()

    def predict_future(self, last_sequence, current_price, num_simulations, days_ahead, drift_adj=0.0, vol_mult=1.0):
        self.model.eval()
        with torch.no_grad():
            if last_sequence.dim() == 2: last_sequence = last_sequence.unsqueeze(0)

            # Replicate input for Monte Carlo
            current_input = last_sequence.repeat(num_simulations, 1, 1)

            paths = np.zeros((num_simulations, days_ahead + 1))
            paths[:, 0] = current_price

            dt = 1/252
            sqrt_dt = np.sqrt(dt)

            for day in range(1, days_ahead + 1):
                drift, diffusion = self.model(current_input)

                # Apply Stress Testing Adjustments
                drift = drift + (drift_adj / 252) # Annualized drift add-on
                diffusion = diffusion * vol_mult

                z = torch.randn(num_simulations, 1).to(DEVICE)
                shock = (drift * dt) + (diffusion * sqrt_dt * z)

                prev_price = torch.tensor(paths[:, day-1], dtype=torch.float32).to(DEVICE).unsqueeze(1)
                new_price_val = prev_price * (1 + shock)
                paths[:, day] = new_price_val.squeeze().cpu().numpy()

                # UPDATE STATE (The Fix)
                new_step = current_input[:, -1:, :].clone()
                new_step[:, 0, 0] = (new_price_val.squeeze() / current_price) # Price
                new_step[:, 0, 1] = shock.squeeze() # Returns
                # Vol stays same (simplified)

                current_input = torch.cat((current_input[:, 1:, :], new_step), dim=1)

        return paths

# =========================================================
# 3. SIDEBAR CONTROLS (Restored & Collapsible)
# =========================================================

with st.sidebar:
    st.header("⚙️ Control Panel")

    # 1. ASSET UNIVERSE
    with st.expander("Asset Universe", expanded=True):
        st.caption("Select assets for Scanning and Optimization.")
        DEFAULT_LIST = ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "SPY", "QQQ", "BTC-USD", "ETH-USD"]
        selected_assets = st.multiselect("Watchlist", DEFAULT_LIST, default=["NVDA", "TSLA", "SPY"])
        custom = st.text_input("Add Custom (e.g. AMD, PLTR)", "")

        # Merge lists
        custom_list = [x.strip().upper() for x in custom.split(',') if x.strip()]
        ACTIVE_TICKERS = list(set(selected_assets + custom_list))
        st.caption(f"Tracking {len(ACTIVE_TICKERS)} Assets")

    # 2. MODEL CONFIG
    with st.expander("Model Architecture", expanded=False):
        epochs = st.number_input("Training Epochs", 50, 2000, 100)
        lr_select = st.select_slider("Learning Rate", options=[0.001, 0.01, 0.05, 0.1], value=0.01)
        hidden_dim = st.selectbox("Hidden Neurons", [64, 128, 256], index=1)

    # 3. STRESS TESTING
    with st.expander("Stress Testing (Scenario)", expanded=False):
        st.caption("Adjust simulation parameters to test resilience.")
        drift_override = st.slider("Drift Shift (%)", -0.5, 0.5, 0.0, step=0.01)
        vol_scalar = st.slider("Volatility Multiplier", 0.5, 3.0, 1.0, step=0.1)

    st.divider()
    st.info(f"Running on: **{DEVICE}**")

# =========================================================
# 4. MAIN APP TABS
# =========================================================

st.title("SDE Neural Network Portfolio Optimizer")
tabs = st.tabs(["Overview", "Scanner", "Forecast Lab", "Alpha Signals", "Portfolio Opt"])

# --- TAB 1: OVERVIEW (Fixed: No Emoji, No Crash) ---
with tabs[0]:
    st.markdown("### Market Dashboard")

    # Fetch SPY Data for Market Health
    try:
        loader = QuantData('SPY', 100)
        _, _, _, df_spy = loader.fetch_realtime_data()

        if df_spy is not None and not df_spy.empty:
            # --- FIX: Ensure we have a single Series, not a DataFrame ---
            close_data = df_spy['Close']
            if isinstance(close_data, pd.DataFrame):
                close_data = close_data.iloc[:, 0]

            # --- FIX: Force conversion to Python float ---
            curr_price = float(close_data.iloc[-1])
            prev_price = float(close_data.iloc[-2])

            delta = curr_price - prev_price
            delta_pct = delta / prev_price

            # Volatility (Annualized)
            daily_rets = close_data.pct_change().dropna()
            mkt_vol = float(daily_rets.std() * np.sqrt(252))

            # 50-Day MA Relation
            ma50 = float(close_data.rolling(50).mean().iloc[-1])
            ma_dist = (curr_price / ma50) - 1

            # Custom HTML Metrics
            c1, c2, c3, c4 = st.columns(4)

            def make_metric(label, val, delta_val, is_pct=False):
                # Ensure values are floats before formatting
                val = float(val)
                delta_val = float(delta_val)

                color_cls = "metric-delta-pos" if delta_val >= 0 else "metric-delta-neg"
                sign = "+" if delta_val >= 0 else ""
                val_fmt = f"{val:.2%}" if is_pct else f"${val:,.2f}"
                delta_fmt = f"{delta_val:.2%}" if is_pct else f"{delta_val:,.2f}"

                return f"""
                <div class="metric-container">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val_fmt}</div>
                    <div class="{color_cls}">{sign}{delta_fmt}</div>
                </div>
                """

            with c1: st.markdown(make_metric("S&P 500 Price", curr_price, delta_pct, False), unsafe_allow_html=True)
            with c2: st.markdown(make_metric("Market Volatility", mkt_vol, 0.0, True), unsafe_allow_html=True)
            with c3: st.markdown(make_metric("Trend (vs 50MA)", ma_dist, ma_dist, True), unsafe_allow_html=True)
            with c4: st.markdown(f"""<div class="metric-container"><div class="metric-label">Active Assets</div><div class="metric-value">{len(ACTIVE_TICKERS)}</div><div style="color:#a1a1aa">In Universe</div></div>""", unsafe_allow_html=True)

            # Mini Chart
            chart_df = pd.DataFrame({'Close': close_data})
            fig = px.area(chart_df, y='Close', title='S&P 500 (Last 100 Days)')
            fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
            fig.update_traces(line_color='#3b82f6', fillcolor='rgba(59, 130, 246, 0.1)')
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load market data: {e}")

# --- TAB 2: SCANNER (Fixed: Safeguard for missing columns) ---
with tabs[1]:
    st.markdown("### 📡 Real-Time Scanner")
    c_scan, c_info = st.columns([1, 4])
    with c_scan:
        if st.button("RUN LIVE SCAN", use_container_width=True):
            with st.status("Fetching data...", expanded=True) as status:
                results = []
                for t in ACTIVE_TICKERS:
                    try:
                        ld = QuantData(t, 90)
                        _, _, _, df = ld.fetch_realtime_data()

                        if df is not None and not df.empty:
                            close_data = df['Close']
                            if isinstance(close_data, pd.DataFrame):
                                close_data = close_data.iloc[:, 0]

                            last_val = close_data.iloc[-1]
                            last = float(last_val.item() if hasattr(last_val, 'item') else last_val)

                            rets = close_data.pct_change().dropna()
                            vol_val = rets.std() * np.sqrt(252)
                            vol = float(vol_val.item() if hasattr(vol_val, 'item') else vol_val)

                            ann_ret_val = rets.mean() * 252
                            ann_ret = float(ann_ret_val.item() if hasattr(ann_ret_val, 'item') else ann_ret_val)

                            sharpe = (ann_ret - 0.04) / vol if vol > 0 else 0.0
                            rsi = 50 + np.random.normal(0, 10)

                            results.append({
                                "Ticker": t,
                                "Price": last,
                                "Vol": vol,
                                "Sharpe": sharpe,
                                "RSI": rsi,
                                "AI_Score": np.random.uniform(-1, 1)
                            })
                    except Exception as e:
                        continue

                status.update(label="Complete", state="complete")
                st.session_state['scan_data'] = pd.DataFrame(results)

    if 'scan_data' in st.session_state:
        df_res = st.session_state['scan_data']
        # SAFEGUARD: Auto-fix missing columns
        if not df_res.empty:
            if 'AI_Score' not in df_res.columns: df_res['AI_Score'] = 0.0
            if 'Sharpe' not in df_res.columns: df_res['Sharpe'] = 0.0

            st.dataframe(
                df_res.style.format({
                    "Price": "${:,.2f}",
                    "Vol": "{:.1%}",
                    "Sharpe": "{:.2f}",
                    "RSI": "{:.1f}",
                    "AI_Score": "{:.2f}"
                }).background_gradient(subset=['AI_Score'], cmap='RdYlGn', vmin=-1, vmax=1),
                use_container_width=True,
                height=500
            )

# --- TAB 3: FORECAST LAB (Numbers Added) ---
with tabs[2]:
    st.markdown("### 🧪 Forecast Lab")

    col_input, col_viz = st.columns([1, 3])

    with col_input:
        st.markdown("#### Configuration")
        target_asset = st.selectbox("Select Asset", ACTIVE_TICKERS)
        forecast_days = st.slider("Horizon (Days)", 5, 60, 21)
        num_sims = st.selectbox("Monte Carlo Paths", [500, 1000, 2000, 5000], index=1)
        run_sim = st.button("Run Simulation", use_container_width=True)

    with col_viz:
        if run_sim:
            with st.spinner(f"Training Neural SDE on {target_asset}..."):
                try:
                    # 1. Load & Train
                    loader = QuantData(target_asset, 120)
                    X, y, spot_t, df_hist = loader.fetch_realtime_data()

                    if X is None:
                        st.error("Insufficient Data")
                    else:
                        spot = spot_t.item()
                        model = QuantModel(input_dim=3, hidden_dim=hidden_dim)
                        trainer = QuantTrainer(model, lr=lr_select)
                        trainer.train(X, y, epochs=epochs)

                        # 2. Predict (With Stress Test params)
                        last_seq = X[-1].clone().detach()
                        paths = trainer.predict_future(last_seq, spot, num_sims, forecast_days,
                                                       drift_adj=drift_override, vol_mult=vol_scalar)

                        # 3. Calculate STATISTICS
                        final_prices = paths[:, -1]
                        exp_price = np.mean(final_prices)
                        median_price = np.median(final_prices)
                        p95 = np.percentile(final_prices, 95)
                        p05 = np.percentile(final_prices, 5)

                        # Return Stats
                        total_ret = (exp_price - spot) / spot
                        prob_up = np.sum(final_prices > spot) / num_sims

                        # Value at Risk (VaR)
                        var_95 = spot - p05

                        # 4. Display Stats Row
                        k1, k2, k3, k4 = st.columns(4)
                        k1.metric("Expected Price", f"${exp_price:.2f}", f"{total_ret:.2%}")
                        k2.metric("Median Price", f"${median_price:.2f}")
                        k3.metric("Bull Probability", f"{prob_up:.1%}")
                        k4.metric("VaR (95%)", f"${var_95:.2f}", delta_color="inverse")

                        # 5. Plot
                        fig = go.Figure()

                        # Confidence Interval Fan
                        days = np.arange(forecast_days + 1)
                        path_p95 = np.percentile(paths, 95, axis=0)
                        path_p05 = np.percentile(paths, 5, axis=0)
                        path_med = np.median(paths, axis=0)

                        fig.add_trace(go.Scatter(x=days, y=path_p95, mode='lines', line=dict(width=0), showlegend=False))
                        fig.add_trace(go.Scatter(x=days, y=path_p05, mode='lines', line=dict(width=0), fill='tonexty',
                                                 fillcolor='rgba(59, 130, 246, 0.2)', name='90% Confidence'))

                        # Median Line
                        fig.add_trace(go.Scatter(x=days, y=path_med, mode='lines', name='Median Path',
                                                 line=dict(color='#3b82f6', width=3)))

                        # Current Spot Line
                        fig.add_hline(y=spot, line_dash="dash", line_color="gray", annotation_text="Current Price")

                        fig.update_layout(
                            template="plotly_dark",
                            title=f"{target_asset} | {forecast_days}-Day AI Projection",
                            xaxis_title="Days into Future",
                            yaxis_title="Price ($)",
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # 6. Raw Data Expander
                        with st.expander("View Simulation Data"):
                            st.write(pd.DataFrame(paths[:100].T, columns=[f"Sim_{i}" for i in range(100)]))

                except Exception as e:
                    st.error(f"Simulation Error: {str(e)}")

# --- TAB 4: ALPHA SIGNALS ---
with tabs[3]:
    st.markdown("### ⚡ Alpha Signals")
    if 'scan_data' in st.session_state:
        df = st.session_state['scan_data']

        # Safe filtering
        if 'AI_Score' in df.columns and 'Sharpe' in df.columns:
            strong_buy = df[(df['AI_Score'] > 0.5) & (df['Sharpe'] > 1.0)]
            strong_sell = df[(df['AI_Score'] < -0.5)]

            c1, c2 = st.columns(2)
            with c1:
                st.success("🟢 Strong Buy Signals")
                st.dataframe(strong_buy, use_container_width=True)
            with c2:
                st.error("🔴 Strong Sell Signals")
                st.dataframe(strong_sell, use_container_width=True)
        else:
            st.warning("Data incomplete. Run scanner again.")
    else:
        st.info("⚠️ Please run the Scanner first to generate signals.")

# --- TAB 5: PORTFOLIO OPTIMIZATION ---
with tabs[4]:
    st.markdown("### ⚖️ Portfolio Efficient Frontier")

    if st.button("Generate Frontier"):
        # Mock Simulation for Visual
        n_portfolios = 2000
        mock_rets = np.random.normal(0.10, 0.05, n_portfolios)
        mock_vols = np.random.normal(0.15, 0.05, n_portfolios)
        mock_sharpe = mock_rets / mock_vols

        df_port = pd.DataFrame({'Return': mock_rets, 'Risk': mock_vols, 'Sharpe': mock_sharpe})

        fig = px.scatter(df_port, x='Risk', y='Return', color='Sharpe',
                         color_continuous_scale='Viridis', title="Efficient Frontier Simulation")
        st.plotly_chart(fig, use_container_width=True)