#!/usr/bin/env python3
"""
Configuration settings for the Insider Trading Detection System
"""

import os
from datetime import datetime, timedelta

# API Configuration
API_KEYS = {
    'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY', 'demo'),
    'finnhub': os.getenv('FINNHUB_API_KEY', 'demo'),
    'polygon': os.getenv('POLYGON_API_KEY', 'demo')
}

# Application Configuration
CONFIG = {
    'app': {
        'debug': os.getenv('DEBUG', 'False').lower() == 'true',
        'host': '0.0.0.0',
        'port': 5000,
        'secret_key': os.getenv('SECRET_KEY', 'insider-trading-detection-key-2024')
    },
    'data': {
        'symbols': [
            'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX',
            'ORCL', 'CRM', 'ADBE', 'PYPL', 'INTC', 'AMD', 'UBER', 'SPOT'
        ],
        'start_date': (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'),
        'end_date': datetime.now().strftime('%Y-%m-%d'),
        'cache_duration': 3600  # Cache data for 1 hour
    },
    'features': {
        'volume_window': 20,
        'volatility_window': 30,
        'momentum_window': 10,
        'rsi_window': 14,
        'bollinger_window': 20,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9
    },
    'detection': {
        'contamination': 0.05,
        'n_neighbors': 20,
        'anomaly_threshold': 2.5,
        'volume_threshold': 3.0,
        'price_threshold': 0.05,
        'ensemble_weights': {
            'isolation_forest': 0.3,
            'local_outlier_factor': 0.25,
            'statistical': 0.25,
            'ml_classifier': 0.2
        }
    },
    'events': {
        'pre_event_window': 30,
        'post_event_window': 5,
        'correlation_threshold': 0.7
    },
    'ml': {
        'test_size': 0.3,
        'cv_folds': 5,
        'random_state': 42,
        'models': {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6
            },
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 1000
            }
        }
    },
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file': 'insider_trading_detection.log'
    }
}

# Risk thresholds for alerts
RISK_THRESHOLDS = {
    'LOW': 1.5,
    'MEDIUM': 2.5,
    'HIGH': 4.0,
    'CRITICAL': 6.0
}

# Visualization settings
VIZ_CONFIG = {
    'colors': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ff7f0e',
        'info': '#17a2b8',
        'anomaly': '#dc3545',
        'normal': '#28a745'
    },
    'chart_height': 400,
    'chart_width': 800
}
