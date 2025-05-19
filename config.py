# config.py
EXCHANGE_CONFIG = {
    'id': 'binance',
    'apiKey': '',  # Leave empty for public data
    'secret': '',  # Leave empty for public data
    'timeout': 30000,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True
    }
}