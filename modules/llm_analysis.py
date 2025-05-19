import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

def generate_market_narrative(df, recommendation, model="llama3"):
    price = df['close'].iloc[-1]
    indicators = {
        "RSI": df['RSI'].iloc[-1],
        "MACD": df['MACD'].iloc[-1],
        "Volume": df['volume'].iloc[-1]
    }

    prompt = f"""Analyze this crypto market situation:
Price: {price}
Indicators: {indicators}
Recommendation: {recommendation}

Provide a concise 5-bullet analysis focusing on:
- Trend strength
- Key support/resistance
- Volume confirmation
- Risk factors
- Suggested action"""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()['message']['content']