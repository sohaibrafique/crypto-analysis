import plotly.graph_objs as go
import numpy as np

def calculate_support_resistance(df, window=5):
    support_levels = []
    resistance_levels = []

    for i in range(window, len(df) - window):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        if high == max(df['high'].iloc[i - window:i + window]):
            resistance_levels.append(high)
        if low == min(df['low'].iloc[i - window:i + window]):
            support_levels.append(low)

    # Return one dominant support and resistance level
    support = min(support_levels) if support_levels else None
    resistance = max(resistance_levels) if resistance_levels else None
    return {'support': support, 'resistance': resistance}

def plot_support_resistance(df, levels):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="Price"
    ))

    if levels['support']:
        fig.add_hline(
            y=levels['support'], line_dash="dot", line_color="green",
            annotation_text="Support", annotation_position="bottom right"
        )
    if levels['resistance']:
        fig.add_hline(
            y=levels['resistance'], line_dash="dot", line_color="red",
            annotation_text="Resistance", annotation_position="top right"
        )

    import streamlit as st
    st.plotly_chart(fig, use_container_width=True)