import time
import streamlit as st
import plotly.graph_objects as go
from data.fetcher import fetch_ohlcv
from analysis.indicators import calculate_technical_indicators
from analysis.trend_analyzer import TrendAnalyzer
from analysis.signal_generator import SignalGenerator
from utils.formatters import format_crypto_value

# test_values = [28453.87623, 5.678, 0.002356, 1200000]
# for val in test_values:
#     print(f"{val:>12,.6f} → {format_crypto_value(val)}")

st.set_page_config(
    layout="wide", 
    page_title="Crypto Price Action Pro",
    page_icon="📊"
)

# Custom CSS for better visuals
st.markdown("""
<style>
    .recommendation-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 1.1em;
    }
    .bullish {
        background-color: #e6f7e6;
        border-left: 5px solid #2ecc71;
    }
    .bearish {
        background-color: #ffebee;
        border-left: 5px solid #e74c3c;
    }
    .neutral {
        background-color: #e3f2fd;
        border-left: 5px solid #3498db;
    }
    .indicator-card {
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

def format_dataframe(df):
    """Apply consistent formatting to display DataFrame"""
    formatted_df = df.copy()
    
    # Price columns
    price_cols = ['open', 'high', 'low', 'close', 'support', 'resistance']
    for col in price_cols:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: format_crypto_value(x, is_price=True))
    
    # Volume columns
    vol_cols = ['volume', 'volume_ma20']
    for col in vol_cols:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: format_crypto_value(x, is_price=False))
    
    return formatted_df

def generate_recommendation(signal_strength, current_price, support, resistance):
    """Generate detailed trading recommendation with proper formatting"""
    # Format all price values
    current_fmt = format_crypto_value(current_price)
    support_fmt = format_crypto_value(support)
    resistance_fmt = format_crypto_value(resistance)
    
    # Calculate percentages
    risk_pct = (current_price-support)/current_price*100
    reward_pct = (resistance-current_price)/current_price*100
    
    if signal_strength > 0.7:
        return {
            "sentiment": "Strong Buy",
            "reason": f"Price above key levels with strong momentum (Current: ${current_fmt})",
            "class": "bullish",
            "action": [
                f"Entry: ${current_fmt}",
                f"Stop Loss: ${support_fmt} ({risk_pct:.1f}% below)",
                f"Take Profit: ${resistance_fmt} ({reward_pct:.1f}% above)"
            ]
        }
    elif signal_strength < -0.7:
        return {
            "sentiment": "Strong Sell",
            "reason": f"Price below key levels with downward momentum (Current: ${current_fmt})",
            "class": "bearish",
            "action": [
                f"Entry: ${current_fmt}",
                f"Stop Loss: ${resistance_fmt} ({reward_pct:.1f}% above)",
                f"Take Profit: ${support_fmt} ({risk_pct:.1f}% below)"
            ]
        }
    else:
        return {
            "sentiment": "Neutral",
            "reason": "Market in consolidation phase - Wait for clearer signal\n\n"
            "Monitor key levels:\n\n",
            "class": "neutral",
            "action": [
                f"Support: ${support_fmt}",
                f"Resistance: ${resistance_fmt}"
            ]
        }

def main():

    st.title("📊 Advanced Crypto Price Action Analysis")
    st.markdown("""
    Real-time market analysis using:
    - Support/Resistance Levels
    - EMA Trends
    - Volume Analysis
    - Candlestick Patterns
    """)
        
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        symbol = st.selectbox("Pair", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"])
        timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"])
        analysis_depth = st.select_slider("Analysis Depth", ["Basic", "Standard", "Advanced"], value="Standard")
    
    try:
        st.write("Fetching data...")
        df = fetch_ohlcv(symbol, timeframe)
        # st.write(f"Initial data: {len(df)} rows, cols: {df.columns.tolist()}")
        st.write("Calculating indicators...")
        df = calculate_technical_indicators(df)
        # st.write(f"Post-indicator cols: {df.columns.tolist()}")
        
        df = TrendAnalyzer().detect_market_regime(df)
        # st.write("Post-regime columns:", df.columns.tolist())

        # Verify all required columns exist
        required_cols = ['regime', 'support', 'resistance', 'ema20']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        signals = SignalGenerator().generate(df)

        # Show sample of critical columns
        st.write("Indicator samples:", df.applymap(format_crypto_value)[['close', 'ema20', 'support', 'resistance']].tail())
        
        current_candle = df.iloc[-1]
        
        # Market Overview Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${format_crypto_value(current_candle['close'])}")
        with col2:
            change_pct = (current_candle['close'] - df.iloc[-2]['close'])/df.iloc[-2]['close']*100
            st.metric("24h Change", f"{change_pct:.2f}%", 
                    "↑ Bullish" if change_pct > 0 else "↓ Bearish")
        with col3:
            st.metric("Market Regime", current_candle['regime'].capitalize())
        
        # Main Chart
        fig = go.Figure()
        
        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ))
        
        # Indicators
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['ema20'],
            line=dict(color='blue', width=1.5),
            name='EMA 20'
        ))
        
        # Support/Resistance
        fig.add_hline(y=current_candle['support'], 
                    line=dict(color='green', dash='dot'),
                    annotation_text=f"Support: ${format_crypto_value(current_candle['support'])}")
        fig.add_hline(y=current_candle['resistance'], 
                    line=dict(color='red', dash='dot'),
                    annotation_text=f"Resistance: ${format_crypto_value(current_candle['resistance'])}")
        
        fig.update_layout(
            title=f"{symbol} Price Action Analysis - {timeframe}",
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation Engine
        signal_strength = (current_candle['close'] - current_candle['ema20'])/current_candle['ema20']
        recommendation = generate_recommendation(
            signal_strength,
            current_candle['close'],
            current_candle['support'],
            current_candle['resistance']
        )
        
        st.markdown(f"""
        <div class="recommendation-box {recommendation['class']}">
            <h3>💰 Trading Recommendation: {recommendation['sentiment']}</h3>
            <p>{recommendation['reason']}</p>
            <ul>
                {''.join([f'<li>{item}</li>' for item in recommendation['action']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed Analysis
        with st.expander("🔍 Detailed Market Analysis", expanded=True):
            tab1, tab2, tab3 = st.tabs(["Trend Analysis", "Key Levels", "Volume Analysis"])
            
            with tab1:
                st.subheader("Trend Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("EMA20", f"${format_crypto_value(current_candle['ema20'])}", 
                            f"{(current_candle['close']-current_candle['ema20'])/current_candle['ema20']*100:.2f}% from price")
                with col2:
                    st.metric("EMA50", f"${format_crypto_value(current_candle['ema50'])}", 
                            f"{(current_candle['close']-current_candle['ema50'])/current_candle['ema50']*100:.2f}% from price")

            with tab2:
                st.subheader("Key Levels")
                st.markdown(f"""
                - Strong Support: ${format_crypto_value(df['support'].min())}
                - Current Support: ${format_crypto_value(current_candle['support'])}
                - Current Resistance: ${format_crypto_value(current_candle['resistance'])}
                - Strong Resistance: ${format_crypto_value(df['resistance'].max())}
                """)
                
                price_range = current_candle['resistance'] - current_candle['support']
                st.metric("Price Range", 
                        f"${format_crypto_value(price_range)}", 
                        f"{(price_range)/current_candle['close']*100:.2f}% of price")

            with tab3:
                st.subheader("Volume Analysis")
                st.metric("Current Volume", format_crypto_value(current_candle['volume'], is_price=False))
                st.metric("Volume vs Average", 
                        f"{current_candle['volume']/current_candle['volume_ma20']:.2f}x",
                        "Above average" if current_candle['volume'] > current_candle['volume_ma20'] else "Below average")
    

        with st.expander("📁 View Raw Data"):
            st.dataframe(
                format_dataframe(df.applymap(format_crypto_value).tail(20).sort_index(ascending=False)),
                column_config={
                    "open": "Open",
                    "high": "High", 
                    "low": "Low",
                    "close": "Close",
                    "volume": st.column_config.NumberColumn(
                        "Volume",
                        format="%.2f"
                    )
                }
            )



    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        if 'df' in locals():
            st.write("Available columns:", df.columns.tolist())
            st.write("Data sample:", df.tail())

if __name__ == "__main__":
    main()