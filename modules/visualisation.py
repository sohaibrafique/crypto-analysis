import plotly.graph_objects as go
import streamlit as st
import pandas as pd

def plot_market_structure(df):
    """Visualize higher highs/lower lows"""
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['close'],
        name='Price',
        line=dict(color='royalblue')
    ))
    
    # Highlight higher highs
    hh_points = df[df['higher_highs']]
    fig.add_trace(go.Scatter(
        x=hh_points.index,
        y=hh_points['high'],
        mode='markers',
        name='Higher High',
        marker=dict(color='green', size=10)
    ))
    
    # Highlight lower lows
    ll_points = df[df['lower_lows']]
    fig.add_trace(go.Scatter(
        x=ll_points.index,
        y=ll_points['low'],
        mode='markers',
        name='Lower Low',
        marker=dict(color='red', size=10)
    ))
    
    fig.update_layout(
        title='Market Structure Analysis',
        yaxis_title='Price',
        hovermode='x unified'
    )
    
    return fig

def plot_price_chart(df, show_ema=False):
    """Plot candlestick chart with optional EMA"""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    required_cols = ['open', 'high', 'low', 'close']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
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
    
    # EMA if requested
    if show_ema and 'EMA20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA20'],
            line=dict(color='blue', width=1),
            name='EMA 20'
        ))
    
    fig.update_layout(
        title='Price Action',
        xaxis_rangeslider_visible=False,
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_volume_chart(df):
    """Plot volume chart with moving average"""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    if 'volume' not in df.columns:
        raise ValueError("DataFrame missing 'volume' column")
    
    fig = go.Figure()
    
    # Volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        name='Volume',
        marker_color='rgba(100, 150, 200, 0.6)'
    ))
    
    # Volume MA if available
    if 'volume_ma20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['volume_ma20'],
            line=dict(color='orange', width=1.5),
            name='20-period Volume MA'
        ))
    
    fig.update_layout(
        title='Trading Volume',
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)