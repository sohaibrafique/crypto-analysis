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