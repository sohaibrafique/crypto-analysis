price_action/
├── data/                  # Data operations
│   ├── fetcher.py         # Enhanced data fetching with caching
│   ├── database.py        # Database manager (new)
│   └── cache.db           # SQLite database
├── analysis/              # Core analysis
│   ├── indicators.py      # Technical indicators
│   ├── signals.py         # Trading signals
│   └── trend_analyzer.py  # Trend analysis (new)
├── backtesting/           # Backtesting suite (new)
│   ├── engine.py          # Backtesting core
│   └── metrics.py         # Performance metrics
├── visualization/         # Visualization (new)
│   ├── charting.py        # Plotting functions
│   └── dashboard/         # Streamlit components
├── config.py              # Central configuration
├── app.py                 # Main Streamlit app
└── README.md              # Project documentation