import streamlit as st
import pandas as pd
from datetime import datetime
from modules.data_fetcher import fetch_data_with_retry
from modules.signals import generate_combined_strategy_signals
from modules.technical_indicators import compute_indicators
from modules.patterns import detect_patterns
from modules.visualisation import plot_price_chart, plot_volume_chart, plot_market_structure
from modules.support_resistance import calculate_support_resistance, plot_support_resistance
from datetime import datetime, timedelta, timezone
import plotly.express as px
from modules.risk_management import calculate_risk_parameters
from modules.llm_analysis import generate_market_narrative
from modules.backtester import backtest_strategy

def multi_timeframe_analysis(base_df, higher_tf_df):
    """Combine insights from different timeframes"""
    analysis = {}
    
    # Trend Alignment
    base_trend = base_df['EMA20'].iloc[-1] > base_df['EMA50'].iloc[-1]
    higher_trend = higher_tf_df['EMA20'].iloc[-1] > higher_tf_df['EMA50'].iloc[-1]
    analysis['trend_alignment'] = 'Aligned' if base_trend == higher_trend else 'Divergence'
    
    # Volume Comparison
    base_volume = base_df['volume'].mean()
    higher_volume = higher_tf_df['volume'].mean()
    analysis['volume_ratio'] = base_volume / higher_volume
    
    return analysis

def format_price(value):
    """Format price values with appropriate decimal places"""
    if value > 10:
        return f"{value:,.0f}"
    elif value > 1:
        return f"{value:,.2f}"
    return f"{value:.4f}"

def format_crypto_value(value, is_price=True):
    """
    Smart formatting for crypto values:
    - Prices > 10: round to nearest 10
    - Prices ≤ 10: 2 decimal places
    - Always add thousands separators
    - Special handling for small altcoin prices
    """
    try:
        value = float(value)
        if is_price:
            if value > 10:
                rounded = round(value / 10) * 10  # Nearest 10
            elif value > 0.1:
                rounded = round(value, 2)  # 2 decimals for mid-range
            else:
                rounded = round(value, 4)  # 4 decimals for tiny values
            
            # Format with commas and remove .0 if unnecessary
            formatted = "{:,.{}f}".format(rounded, 0 if rounded.is_integer() and value > 10 else (4 if value < 0.1 else 2))
            return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted
        else:
            # For non-price values (volumes, etc)
            return "{:,.2f}".format(round(value, 2))
    except:
        return str(value)  # Fallback for non-numeric

def get_level(level_list, idx, fallback):
    try:
        if isinstance(level_list[idx], (list, tuple)):
            return level_list[idx][1]
        else:
            return level_list[idx]
    except:
        return fallback

def generate_recommendation(current_candle, df, levels):
    """Generate trading recommendations with detailed reasoning"""
    current_price = current_candle['close']
    support = get_level(levels.get('support', []), 0, current_price * 0.98)
    resistance = get_level(levels.get('resistance', []), 0, current_price * 1.02)
    ema20 = current_candle['EMA20']
    
    # Formatting helper
    fmt = lambda x: format_crypto_value(x)
    
    # Core analysis logic
    analysis = {
        "price_position": "",
        "key_levels": "",
        "volume_analysis": "",
        "pattern_analysis": "",
        "action": "",
        "reason": []
    }
    
    # 1. Price position analysis
    if current_price > resistance * 0.99:
        analysis["price_position"] = "Price is testing resistance"
        analysis["reason"].append("Asset is trading at the upper boundary of its recent range")
    elif current_price < support * 1.01:
        analysis["price_position"] = "Price is testing support"
        analysis["reason"].append("Asset is trading at the lower boundary of its recent range")
    else:
        analysis["price_position"] = "Price is in mid-range"
        analysis["reason"].append("Asset is trading between key support and resistance levels")

    # 2. Trend analysis
    trend_strength = (current_price - ema20)/ema20 * 100
    if trend_strength > 2:
        analysis["reason"].append(f"Strong uptrend (Price {fmt(trend_strength)}% above EMA20)")
    elif trend_strength < -2:
        analysis["reason"].append(f"Strong downtrend (Price {fmt(abs(trend_strength))}% below EMA20)")

    # 3. Volume analysis
    volume_ratio = current_candle['volume'] / current_candle['volume_ma20']
    if volume_ratio > 1.5:
        analysis["volume_analysis"] = "High volume activity detected"
        analysis["reason"].append("Significant trading volume supporting recent price moves")
    elif volume_ratio < 0.8:
        analysis["volume_analysis"] = "Low volume caution"
        analysis["reason"].append("Low trading volume - potential lack of conviction in price moves")

    # 4. Pattern detection
    patterns = []
    if current_candle.get('bullish_engulfing', False):
        patterns.append("Bullish Engulfing Pattern")
    if current_candle.get('bearish_pin_bar', False):
        patterns.append("Bearish Pin Bar Pattern")
    
    if patterns:
        analysis["pattern_analysis"] = f"Notable candlestick pattern: {', '.join(patterns)}"
        analysis["reason"].append(f"Recent price action shows {patterns[0]} formation")

    # 5. Actionable recommendations
    action_plan = []
    if current_price > resistance * 0.99 and volume_ratio > 1.2:
        action_plan.append({
            "direction": "Buy",
            "condition": f"Price sustains above {fmt(resistance)}",
            "target": fmt(resistance * 1.02),
            "stop": fmt(support)
        })
    elif current_price < support * 1.01 and volume_ratio > 1.2:
        action_plan.append({
            "direction": "Sell",
            "condition": f"Price breaks below {fmt(support)}",
            "target": fmt(support * 0.98),
            "stop": fmt(resistance)
        })
    else:
        action_plan.append({
            "direction": "Wait",
            "condition": "Clear breakout/breakdown",
            "target": fmt(resistance if trend_strength > 0 else support),
            "stop": fmt(support if trend_strength > 0 else resistance)
        })

    # Compile final recommendation
    return {
        "sentiment": "Bullish" if trend_strength > 2 else "Bearish" if trend_strength < -2 else "Neutral",
        "action_plan": action_plan,
        "analysis": analysis,
        "key_levels": {
            "immediate_support": fmt(support),
            "immediate_resistance": fmt(resistance),
            "next_support": fmt(get_level(levels.get('support', []), 1, support)),
            "next_resistance": fmt(get_level(levels.get('resistance', []), 1, resistance))
        }
    }

def color_signal_cells(val):
    """Color coding for signal DataFrame"""
    if val == 'BUY':
        return 'background-color: lightgreen'
    elif val == 'SELL':
        return 'background-color: lightcoral'
    return ''

def validate_dataframe(df):
    """Check DataFrame meets requirements"""
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a DataFrame"
    if df.empty:
        return False, "DataFrame is empty"
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}"
    return True, ""

def show_recommendation(recommendation):
    """Display interactive recommendation with expandable details"""
    with st.expander("🚨 Trading Recommendation", expanded=True):
        st.subheader(f"{recommendation['sentiment']} Bias")
        
        st.markdown("### Key Levels")
        st.write(f"🛑 Immediate Support: {recommendation['key_levels']['immediate_support']}")
        st.write(f"🎯 Immediate Resistance: {recommendation['key_levels']['immediate_resistance']}")
        st.write(f"⏭️ Next Support: {recommendation['key_levels']['next_support']}")
        st.write(f"⏭️ Next Resistance: {recommendation['key_levels']['next_resistance']}")
        
        st.markdown("### Action Plan")
        for action in recommendation['action_plan']:
            st.write(f"**{action['direction']} Signal When:**")
            st.write(f"- 📌 Condition: {action['condition']}")
            st.write(f"- 🎯 Target: {action['target']}")
            st.write(f"- 🛑 Stop Loss: {action['stop']}")
        
        st.markdown("### Analysis Summary")
        for point in recommendation['analysis']['reason']:
            st.write(f"- {point}")

        st.markdown("### Risk Management")
        st.write(f"📉 Suggested Stop Loss: {recommendation['risk_management']['stop_loss_pct']:.2f}%")
        st.write(f"📊 Position Size: {recommendation['risk_management']['position_size']:.2f} units")
        
        st.markdown("### AI Market Narrative")
        st.write(recommendation['narrative'])

def input_section():
    with st.sidebar:
        st.title("Crypto Analysis Settings")
        
        symbol = st.selectbox(
            "Select Pair", 
            ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"],
            index=0
        )
        
        interval = st.selectbox(
            "Timeframe", 
            ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
            index=2
        )
        
        # Default to last 30 days
        default_end = datetime.now(timezone.utc)
        default_start = default_end - timedelta(days=30)
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=default_start.date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=default_end.date()
            )
        
        # Convert to datetime at midnight UTC
        start_dt = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)
        
        if start_dt >= end_dt:
            st.error("End date must be after start date")
            st.stop()
            
        return symbol, interval, start_dt, end_dt

def analysis_section(symbol, interval, start_date, end_date):
    """Handle data fetching and processing with proper error handling"""
    try:
        # Fetch data (will raise exception on failure)
        df = fetch_data_with_retry(symbol, interval, start_date, end_date)
        
        # Process data
        df = compute_indicators(df)
        df = detect_patterns(df)
        
        risk_params = calculate_risk_parameters(df)

        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        return df, risk_params
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        st.info(f"Trying fetching test data instead...")
        
        # Fallback to test data
        try:
            test_start = datetime.now(timezone.utc) - timedelta(days=7)
            test_end = datetime.now(timezone.utc)
            df = fetch_data_with_retry("BTC/USDT", "4h", test_start, test_end)
            st.warning(f"Showing test data (BTC/USDT 4h last 7 days) instead")
            return df
        except Exception as test_error:
            st.error(f"Test data also failed: {str(test_error)}")
            return None

def output_section(df, risk_params):
    """Display analysis with graceful fallbacks"""
    if df is None or df.empty:
        st.warning("No data available for display")
        return

    if 'timestamp' not in df.columns:
        st.warning("Missing 'timestamp' column")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['time'] = df['timestamp']
    df = df.set_index('time')  # <-- Make timestamp the x-axis for all plots

    try:
        tab1, tab2, tab3, tab4 = st.tabs(["Charts", "Patterns", "Signals", "Recommendations"])

        with tab1:
            st.subheader("Price Action")
            plot_price_chart(df)  # Uses df.index (which is now timestamp)

            st.subheader("Trading Volume")
            plot_volume_chart(df)  # Uses df.index (which is now timestamp)

            levels = calculate_support_resistance(df)
            if levels:
                plot_support_resistance(df, levels)
            else:
                st.info("No significant support/resistance levels found")

        with tab2:
            if 'pattern' in df.columns:
                patterns_df = df[df['pattern'].notna()]
                if not patterns_df.empty:
                    st.dataframe(patterns_df[['timestamp', 'pattern']])
                else:
                    st.info("No candlestick patterns detected")
            else:
                st.info("Pattern detection not applied or failed")

        with tab3:
            signals = generate_combined_strategy_signals(df)

            if not signals.empty:
                if 'timestamp' not in signals.columns:
                    signals = signals.reset_index()  # Ensure timestamp is column

                st.dataframe(
                    signals.style.applymap(
                        lambda x: "background-color: lightgreen" if x == "Buy"
                        else "background-color: lightcoral" if x == "Sell"
                        else "",
                        subset=['signal']
                    )
                )
            else:
                st.info("No trading signals generated")

        with tab4:  # Recommendations tab
            if not df.empty:
                recommendation = generate_recommendation(df.iloc[-1], df, levels)
                
                # Add risk parameters
                recommendation['risk_management'] = risk_params
                
                # Add LLM analysis
                recommendation['narrative'] = generate_market_narrative(df, recommendation)
                
                show_recommendation(recommendation)

            with st.expander("Market Structure Analysis"):
                st.plotly_chart(plot_market_structure(df))
                
                st.write("""
                **Pattern Legend:**
                - 🟢 Higher High: Bullish continuation signal
                - 🔴 Lower Low: Bearish continuation signal
                - When combined with volume analysis, these become powerful signals
                """)

    except Exception as e:
        st.error(f"Display error: {str(e)}")
        st.write("Debug info:", df.columns if df is not None else "No DataFrame")

def app():
    st.set_page_config(
        layout="wide",
        page_title="Crypto Analyst Pro",
        page_icon="📈"
    )
    
    st.title("Cryptocurrency Technical Analysis")
    
    # Get inputs
    try:
        symbol, interval, start_date, end_date = input_section()
        
        with st.spinner("Analyzing market data..."):
            df, risk_params = analysis_section(symbol, interval, start_date, end_date)
            
            if df is not None:
                # Add backtesting results
                backtest_results = backtest_strategy(df)
                
                output_section(df, risk_params)
                
                # Show backtesting results in new tab
                with st.expander("Backtesting Results"):
                    st.metric("Strategy Return", f"{backtest_results['return_pct']:.2f}%")
                    st.dataframe(pd.DataFrame(backtest_results['trades']))

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.write("Please try again or contact support")

if __name__ == "__main__":
    app()