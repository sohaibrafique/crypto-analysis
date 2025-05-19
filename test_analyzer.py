import argparse
from binance.client import Client
import pandas as pd
# import ollama
import matplotlib.pyplot as plt
import streamlit as st
import datetime
import plotly.graph_objects as go
import re
import numpy as np
from datetime import date
from pandas import Timestamp
from scipy.signal import argrelextrema
import os

if os.getenv("STREAMLIT_CLOUD") != "1":
    import ollama

def clean_special_characters(text):
    # Remove unwanted special characters but keep basic punctuation
    return re.sub(r'[^a-zA-Z0-9\s.,:\-_$]', '', text)

def fetch_data(pair="ETHUSDT", interval="4h", lookback=1000, start_date=None, end_date=None):
    client = Client(api_key, api_secret, testnet=True)
    client.API_URL = 'https://testnet.binance.vision/api'
    
    # client = Client()
    klines = client.get_klines(symbol=pair, interval=interval, limit=lookback)
    df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", 
                                      "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    # df["close"] = df["close"].astype(float)
    df['open'] = pd.to_numeric(df['open'], errors='coerce')
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df['high'] = pd.to_numeric(df['high'], errors='coerce')
    df['low'] = pd.to_numeric(df['low'], errors='coerce')
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')    
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    if start_date and end_date:
        df = df[(df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)]
    return df

def is_doji(df, threshold=0.1):
    body = abs(df['close'] - df['open'])
    range_ = df['high'] - df['low']
    return (body / range_) < threshold

def is_hammer(df):
    body = abs(df['close'] - df['open'])
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)

    return (lower_wick > 2 * body) & (upper_wick < body)

def is_engulfing(df):
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    curr_open = df['open']
    curr_close = df['close']

    prev_bull = prev_close > prev_open
    prev_bear = prev_open > prev_close
    curr_bull = curr_close > curr_open
    curr_bear = curr_open > curr_close

    bull_engulfing = prev_bear & curr_bull & (curr_close > prev_open) & (curr_open < prev_close)
    bear_engulfing = prev_bull & curr_bear & (curr_open > prev_close) & (curr_close < prev_open)

    return bull_engulfing | bear_engulfing

def detect_dynamic_support_resistance(df, order=3):
    # df['swing_low'] = df['low'][argrelextrema(df['low'].values, np.less_equal, order=order)[0]]
    # df['swing_high'] = df['high'][argrelextrema(df['high'].values, np.greater_equal, order=order)[0]]

    # Initialise columns
    df['swing_low'] = np.nan
    df['swing_high'] = np.nan

    # Get extrema positions
    low_indices = argrelextrema(df['low'].values, np.less_equal, order=order)[0]
    high_indices = argrelextrema(df['high'].values, np.greater_equal, order=order)[0]

    # Assign using iloc (positional)
    df.iloc[low_indices, df.columns.get_loc('swing_low')] = df['low'].iloc[low_indices]
    df.iloc[high_indices, df.columns.get_loc('swing_high')] = df['high'].iloc[high_indices]

    # Get the most recent valid values
    recent_supports = df['swing_low'].dropna().tail(3)
    recent_resistances = df['swing_high'].dropna().tail(3)
    
    support_level = recent_supports.min() if not recent_supports.empty else np.nan
    resistance_level = recent_resistances.max() if not recent_resistances.empty else np.nan

    df['Support'] = support_level
    df['Resistance'] = resistance_level
    return df

def calculate_indicators(df):
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['EMA100'] = df['close'].ewm(span=100, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()

    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()

    df['Volume_MA20'] = df['volume'].rolling(window=20).mean()

    # Standard RSI calculation using EMA
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Candle patterns
    df['Doji'] = is_doji(df)
    df['Hammer'] = is_hammer(df)
    df['Engulfing'] = is_engulfing(df)

    # MACD
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["Signal"]

    # Dynamic Support/Resistance
    df = detect_dynamic_support_resistance(df)

    return df

def analyze_macd(df):
    """Enhanced MACD analysis with cross and divergence detection"""
    last_macd = df["MACD"].iloc[-1]
    last_signal = df["Signal"].iloc[-1]
    last_hist = df["MACD_Hist"].iloc[-1]
    prev_hist = df["MACD_Hist"].iloc[-2]
    
    analysis = []
    
    # Cross analysis
    if last_macd > last_signal and df["MACD"].iloc[-2] <= df["Signal"].iloc[-2]:
        analysis.append("Bullish cross (MACD just crossed above Signal)")
    elif last_macd < last_signal and df["MACD"].iloc[-2] >= df["Signal"].iloc[-2]:
        analysis.append("Bearish cross (MACD just crossed below Signal)")
    
    # Divergence analysis (simplified)
    if (df["close"].iloc[-1] > df["close"].iloc[-3]) and (last_macd < df["MACD"].iloc[-3]):
        analysis.append("Potential bearish divergence (price higher but MACD lower)")
    elif (df["close"].iloc[-1] < df["close"].iloc[-3]) and (last_macd > df["MACD"].iloc[-3]):
        analysis.append("Potential bullish divergence (price lower but MACD higher)")
    
    # Strength analysis
    if last_hist > 0 and last_hist > prev_hist:
        analysis.append("Bullish momentum increasing")
    elif last_hist < 0 and last_hist < prev_hist:
        analysis.append("Bearish momentum increasing")
    
    return " | ".join(analysis) if analysis else "No strong signals"

def rsi_interpretation(rsi_value):
    if rsi_value < 30:
        return "Oversold"
    elif rsi_value > 70:
        return "Overbought"
    else:
        return "Neutral"

def get_llm_recommendation(analysis_text, timeframe, df):
    current_price = float(df['close'].iloc[-1])
    support = float(df['Support'].iloc[-1])
    resistance = float(df['Resistance'].iloc[-1])
    stop_loss = round(support * 0.96, 3)
    take_profit = round(current_price + 2 * (current_price - stop_loss), 2)
    rsi = float(df['RSI'].iloc[-1])
    macd_hist = float(df['MACD_Hist'].iloc[-1])
    volume = int(float(df['volume'].iloc[-1]))

    structured_prompt = f"""
You are an expert crypto market analyst. Based on the provided data and analysis, generate a concise trading recommendation in the following structure:

**Trade Recommendation:**
- Action: [e.g. Speculative Buy, Watch, Hold, Avoid]
- Entry: Around ${current_price:.2f}
- Stop Loss: ${stop_loss:.3f} (below support)
- Take Profit: ${take_profit:.2f}
- Risk/Reward: ~2:1
- Confidence Level: [High, Medium, Low]
- Timeframe: {timeframe}
- Position Size: 4% of capital (adjusted for volatility)
- Validation: Price must hold above ${support:.2f} and MACD line must cross above signal line for confirmation.

**Context:**
- Timeframe: {timeframe}
- Current Price: ${current_price:.2f}
- RSI: {rsi:.1f}
- MACD Histogram: {macd_hist:.4f}
- Volume: {volume:,}
- Support: ${support:.2f}
- Resistance: ${resistance:.2f}
- Additional Notes: {analysis_text.strip()}

Keep it in plain text format only, use colon-separated key-value pairs, no markdown or special characters, and round all prices to 2 decimals except stop loss (3 decimals).
"""

    response = ollama.chat(
        model='llama3',
        messages=[{'role': 'user', 'content': structured_prompt}]
    ).get("message", {}).get("content", "")

    if not re.search(r"Action:\s*(Speculative Buy|Buy|Hold|Avoid|Watch)", response):
        return "Action: Hold\nReason: No strong confirmation from indicators yet."

    return clean_special_characters(response)

def show_dashboard(df, recommendation, pair, timeframe):
    st.title(f"Crypto Analysis Dashboard - {pair} ({timeframe})")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Price", f"${df['close'].iloc[-1]:.2f}")
    with col2:
        st.metric("Trend", "Bullish" if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] else "Bearish")
    
    show_interactive_charts(df, pair)
    
    st.subheader("Trading Recommendation")
    st.markdown(recommendation, unsafe_allow_html=True)
    
    st.subheader("Recent Data")
    st.dataframe(df.tail()[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'EMA20', 'EMA50', 'RSI']])

def validate_recommendation(recommendation, df):
    """Multi-factor validation matrix"""
    conditions = {
        'BUY': [
            df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1],
            df['RSI'].iloc[-1] < 45,
            df['MACD_Hist'].iloc[-1] > 0,
            df['close'].iloc[-1] > df['Support'].iloc[-1]
        ],
        'SELL': [
            df['EMA20'].iloc[-1] < df['EMA50'].iloc[-1],
            df['RSI'].iloc[-1] > 55,
            df['MACD_Hist'].iloc[-1] < 0,
            df['close'].iloc[-1] < df['Resistance'].iloc[-1]
        ]
    }
    
    action = recommendation.split(':')[0].strip()
    required_conditions = conditions.get(action, [True])
    met_conditions = sum(required_conditions)
    
    if met_conditions/len(required_conditions) >= 0.75:
        return "Valid Recommendation"
    else:
        return f"Warning: Only {met_conditions}/{len(required_conditions)} conditions met"

def generate_interactive_report(df, pair="ETHUSDT"):
    fig = go.Figure()

    # Plot price and EMAs
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["EMA20"], mode='lines', name='EMA20'))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["EMA50"], mode='lines', name='EMA50'))

    # Get last support/resistance
    support = df['Support'].iloc[-1]
    resistance = df['Resistance'].iloc[-1]
    band_size = 0.005  # 0.5% band width

    # Add horizontal bands using shapes
    fig.add_shape(type="rect",
        x0=df["timestamp"].iloc[0], x1=df["timestamp"].iloc[-1],
        y0=support * (1 - band_size), y1=support * (1 + band_size),
        fillcolor="green", opacity=0.2, layer="below", line_width=0,
        xref="x", yref="y"
    )

    fig.add_shape(type="rect",
        x0=df["timestamp"].iloc[0], x1=df["timestamp"].iloc[-1],
        y0=resistance * (1 - band_size), y1=resistance * (1 + band_size),
        fillcolor="red", opacity=0.2, layer="below", line_width=0,
        xref="x", yref="y"
    )

    # Dummy traces for legend
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color="green", width=10),
                             name='Support Band'))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                             line=dict(color="red", width=10),
                             name='Resistance Band'))

    fig.update_layout(title=f"{pair} Price with EMAs",
                      xaxis_title="Time", yaxis_title="Price", height=600,
                      hovermode='x unified')
    return fig

def find_ema_crossovers(df):
    """Detect EMA20/50 crossovers"""
    crosses = []
    for i in range(1, len(df)):
        if (df['EMA20'].iloc[i-1] <= df['EMA50'].iloc[i-1] and 
            df['EMA20'].iloc[i] > df['EMA50'].iloc[i]):
            crosses.append({'timestamp': df['timestamp'].iloc[i], 'type': 'bullish'})
        elif (df['EMA20'].iloc[i-1] >= df['EMA50'].iloc[i-1] and 
              df['EMA20'].iloc[i] < df['EMA50'].iloc[i]):
            crosses.append({'timestamp': df['timestamp'].iloc[i], 'type': 'bearish'})
    return crosses

def find_macd_crossovers(df):
    """Detect MACD/Signal line crossovers"""
    crosses = []
    for i in range(1, len(df)):
        if (df['MACD'].iloc[i-1] <= df['Signal'].iloc[i-1] and 
            df['MACD'].iloc[i] > df['Signal'].iloc[i]):
            crosses.append({'timestamp': df['timestamp'].iloc[i], 'type': 'bullish'})
        elif (df['MACD'].iloc[i-1] >= df['Signal'].iloc[i-1] and 
              df['MACD'].iloc[i] < df['Signal'].iloc[i]):
            crosses.append({'timestamp': df['timestamp'].iloc[i], 'type': 'bearish'})
    return crosses

def detect_divergence(df, lookback=14):
    """Simple divergence detection between price and RSI"""
    bullish = bearish = False
    if len(df) > lookback:
        # Price makes lower low but RSI makes higher low
        if (df['close'].iloc[-1] < df['close'].iloc[-lookback] and
            df['RSI'].iloc[-1] > df['RSI'].iloc[-lookback]):
            bullish = True
        # Price makes higher high but RSI makes lower high
        elif (df['close'].iloc[-1] > df['close'].iloc[-lookback] and
              df['RSI'].iloc[-1] < df['RSI'].iloc[-lookback]):
            bearish = True
    return bullish, bearish

def show_interactive_charts(df, pair):
    """Display enhanced interactive charts with technical indicators"""
    with st.container():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"{pair} Price Action")
            fig_price = go.Figure()

            # Add support/resistance zones as bands
            support = df['Support'].iloc[-1]
            resistance = df['Resistance'].iloc[-1]
            band_size = 0.01  # 5% band

            fig_price.add_shape(
                type="rect",
                x0=df["timestamp"].iloc[0], x1=df["timestamp"].iloc[-1],
                y0=support * (1 - band_size), y1=support * (1 + band_size),
                fillcolor="green", opacity=0.15, line_width=0,
                layer="below", xref="x", yref="y"
            )

            fig_price.add_shape(
                type="rect",
                x0=df["timestamp"].iloc[0], x1=df["timestamp"].iloc[-1],
                y0=resistance * (1 - band_size), y1=resistance * (1 + band_size),
                fillcolor="red", opacity=0.15, line_width=0,
                layer="below", xref="x", yref="y"
            )

            # Add price and indicators
            fig_price.add_trace(go.Candlestick(
                x=df["timestamp"],
                open=df["open"], high=df["high"],
                low=df["low"], close=df["close"],
                name='Price'
            ))

            fig_price.add_trace(go.Scatter(
                x=df["timestamp"], y=df["EMA20"],
                mode='lines', name='EMA20',
                line=dict(color='royalblue', width=1.5)
            ))

            fig_price.add_trace(go.Scatter(
                x=df["timestamp"], y=df["EMA50"],
                mode='lines', name='EMA50',
                line=dict(color='orange', width=1.5)
            ))

            fig_price.add_trace(go.Scatter(
                x=df["timestamp"], y=df["Support"],
                line=dict(color='green', width=1, dash='dot'),
                name='Support', opacity=0.7
            ))

            fig_price.add_trace(go.Scatter(
                x=df["timestamp"], y=df["Resistance"],
                line=dict(color='red', width=1, dash='dot'),
                name='Resistance', opacity=0.7
            ))

            fig_price.update_layout(
                xaxis_rangeslider_visible=False,
                height=500,
                hovermode='x unified',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col2:
            st.subheader("Key Metrics")
            st.metric("Current RSI", f"{df['RSI'].iloc[-1]:.1f}", 
                     delta=f"{df['RSI'].iloc[-1] - df['RSI'].iloc[-2]:.1f}",
                     delta_color="inverse")
            st.metric("MACD Hist", f"{df['MACD_Hist'].iloc[-1]:.4f}",
                     delta=f"{df['MACD_Hist'].iloc[-1] - df['MACD_Hist'].iloc[-2]:.4f}")
            st.metric("Support", f"${support:.2f}")
            st.metric("Resistance", f"${resistance:.2f}")

            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            vol_delta = "0%" if df['volume'].iloc[-2] == 0 else \
                        f"{((df['volume'].iloc[-1]/df['volume'].iloc[-2])-1)*100:.1f}%"
            st.metric("Volume", f"{df['volume'].iloc[-1]:,.0f}", delta=vol_delta)

    with st.expander("MACD Analysis", expanded=True):
        fig_macd = go.Figure()

        colors = np.where(df['MACD_Hist'] > 0, 'green', 'red')
        fig_macd.add_trace(go.Bar(
            x=df["timestamp"], y=df["MACD_Hist"],
            name="Histogram", marker_color=colors, opacity=0.6
        ))

        fig_macd.add_trace(go.Scatter(
            x=df["timestamp"], y=df["MACD"],
            name="MACD", line=dict(color='blue', width=1.5)
        ))

        fig_macd.add_trace(go.Scatter(
            x=df["timestamp"], y=df["Signal"],
            name="Signal", line=dict(color='darkorange', width=1.5)
        ))

        fig_macd.add_hline(y=0, line_width=1, line_color="black")
        fig_macd.update_layout(height=300, hovermode='x unified', showlegend=True)
        st.plotly_chart(fig_macd, use_container_width=True)

    with st.expander("RSI Analysis", expanded=True):
        fig_rsi = go.Figure()

        fig_rsi.add_trace(go.Scatter(
            x=df["timestamp"], y=df["RSI"],
            name="RSI", line=dict(color='purple', width=2)
        ))

        fig_rsi.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, layer="below", line_width=0)
        fig_rsi.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, layer="below", line_width=0)
        fig_rsi.add_hline(y=50, line_width=1, line_color="gray")

        if 'detect_divergence' in globals():
            bullish_div, bearish_div = detect_divergence(df)
            if bullish_div:
                fig_rsi.add_annotation(
                    x=df["timestamp"].iloc[-1],
                    y=df["RSI"].iloc[-1],
                    text="Bullish Divergence",
                    showarrow=True, arrowhead=1
                )

        fig_rsi.update_layout(height=300, yaxis_range=[0, 100], hovermode='x unified')
        st.plotly_chart(fig_rsi, use_container_width=True)

if __name__ == "__main__":
    st.sidebar.title("Settings")
    pair = st.sidebar.selectbox("Select Trading Pair", ["ETHUSDT", "BTCUSDT", "NEARUSDT", "ADAUSDT", "ALGOUSDT"])
    timeframe = st.sidebar.selectbox("Select Timeframe", ['3m', '15m', '30m', '1h', '4h', '1d', '1w'])
    
    today = date.today()
    start_date = st.sidebar.date_input("Start Date", today - datetime.timedelta(days=30))
    end_date = st.sidebar.date_input("End Date", today)
    
    if st.sidebar.button("Run Analysis"):
        with st.spinner("Fetching data and analyzing..."):
            df = fetch_data(pair=pair, interval=timeframe, start_date=start_date, end_date=end_date)
            df = calculate_indicators(df)
            
            macd_analysis = analyze_macd(df)
            
            analysis_text = f"""
            Current Price: ${df['close'].iloc[-1]:.2f}
            Trend: {'Bullish' if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] else 'Bearish'}
            RSI: {df['RSI'].iloc[-1]:.2f} ({'Oversold' if df['RSI'].iloc[-1] < 30 else 'Overbought' if df['RSI'].iloc[-1] > 70 else 'Neutral'})
            MACD Analysis: {macd_analysis}
            Support: ${df['Support'].iloc[-1]:.2f}
            Resistance: ${df['Resistance'].iloc[-1]:.2f}
            """
            
            recommendation = get_llm_recommendation(analysis_text, timeframe, df)
            validation = validate_recommendation(recommendation, df)
            
            show_dashboard(df, recommendation, pair, timeframe)
