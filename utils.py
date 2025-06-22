#!/usr/bin/env python3
"""
Utility functions for the Insider Trading Detection System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import json

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def validate_data(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate that DataFrame contains required columns and data."""
    if df is None or df.empty:
        return False
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logging.warning(f"Missing required columns: {missing_columns}")
        return False
    
    # Check for sufficient non-null data
    for col in required_columns:
        if df[col].isna().sum() / len(df) > 0.5:  # More than 50% missing
            logging.warning(f"Column {col} has too many missing values")
            return False
    
    return True

def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean numeric data by handling inf, -inf, and NaN values."""
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with column medians for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isna().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(median_val)
    
    return df

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate percentage returns from price series."""
    return prices.pct_change().fillna(0)

def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate log returns from price series."""
    return np.log(prices / prices.shift(1)).fillna(0)

def calculate_volatility(returns: pd.Series, window: int = 30) -> pd.Series:
    """Calculate rolling volatility."""
    return returns.rolling(window=window).std()

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio."""
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0

def detect_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using Z-score method."""
    z_scores = np.abs(stats.zscore(data.dropna()))
    return pd.Series(z_scores > threshold, index=data.index)

def detect_outliers_iqr(data: pd.Series) -> pd.Series:
    """Detect outliers using Interquartile Range method."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

def normalize_scores(scores: pd.Series) -> pd.Series:
    """Normalize scores to 0-1 range."""
    min_score = scores.min()
    max_score = scores.max()
    if max_score == min_score:
        return pd.Series(0.5, index=scores.index)
    return (scores - min_score) / (max_score - min_score)

def calculate_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix for numeric columns."""
    numeric_data = data.select_dtypes(include=[np.number])
    return numeric_data.corr()

def format_percentage(value: float, decimals: int = 2) -> str:
    """Format number as percentage."""
    return f"{value * 100:.{decimals}f}%"

def format_currency(value: float, currency: str = "$") -> str:
    """Format number as currency."""
    return f"{currency}{value:,.2f}"

def get_risk_level(score: float, thresholds: Dict[str, float]) -> str:
    """Determine risk level based on score and thresholds."""
    if score >= thresholds['CRITICAL']:
        return 'CRITICAL'
    elif score >= thresholds['HIGH']:
        return 'HIGH'
    elif score >= thresholds['MEDIUM']:
        return 'MEDIUM'
    elif score >= thresholds['LOW']:
        return 'LOW'
    else:
        return 'NORMAL'

def create_time_windows(df: pd.DataFrame, window_size: int, step_size: int = 1) -> List[pd.DataFrame]:
    """Create sliding time windows from DataFrame."""
    windows = []
    for i in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[i:i + window_size].copy()
        windows.append(window)
    return windows

def aggregate_anomaly_scores(scores_dict: Dict[str, pd.Series], weights: Dict[str, float]) -> pd.Series:
    """Aggregate multiple anomaly scores using weighted average."""
    if not scores_dict:
        return pd.Series()
    
    # Normalize all scores to 0-1 range
    normalized_scores = {}
    for method, scores in scores_dict.items():
        normalized_scores[method] = normalize_scores(scores)
    
    # Calculate weighted average
    result = pd.Series(0.0, index=list(scores_dict.values())[0].index)
    total_weight = 0
    
    for method, scores in normalized_scores.items():
        weight = weights.get(method, 1.0)
        result += scores * weight
        total_weight += weight
    
    return result / total_weight if total_weight > 0 else result

def export_to_json(data: Any, filename: str) -> bool:
    """Export data to JSON file."""
    try:
        with open(filename, 'w') as f:
            if isinstance(data, pd.DataFrame):
                json.dump(data.to_dict('records'), f, indent=2, default=str)
            else:
                json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        logging.error(f"Error exporting to JSON: {e}")
        return False

def load_from_json(filename: str) -> Optional[Any]:
    """Load data from JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading from JSON: {e}")
        return None

from scipy import stats
