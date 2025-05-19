"""
Self-documenting charting functions
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_price_action(df, signals=None):
    """
    Creates interactive price chart with:
    - Candlesticks
    - Key indicators
    - Trading signals
    
    Parameters:
        df (pd.DataFrame): Must contain OHLC columns
        signals (pd.DataFrame): Optional signals to plot
        
    Returns:
        plotly.Figure
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       row_heights=[0.7, 0.3], vertical_spacing=0.05)
    
    # Price Chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price'
    ), row=1, col=1)
    
    # Volume Chart
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        name='Volume',
        marker_color='rgba(100, 150, 200, 0.6)'
    ), row=2, col=1)
    
    # Add indicators if present
    if 'ema20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['ema20'],
            line=dict(color='blue', width=1.5),
            name='EMA 20'
        ), row=1, col=1)
    
    # Add signals if provided
    if signals is not None:
        buy_signals = signals[signals['signal'] == 1]
        sell_signals = signals[signals['signal'] == -1]
        
        fig.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals['price'],
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color='green'),
            name='Buy Signal'
        ), row=1, col=1)
    
    fig.update_layout(
        title='Price Action Analysis',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig