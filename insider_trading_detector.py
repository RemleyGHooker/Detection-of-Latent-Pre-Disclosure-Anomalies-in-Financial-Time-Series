#!/usr/bin/env python3
"""
Complete Insider Trading Signal Detection System
A comprehensive system for detecting potential insider trading patterns
using anomaly detection and machine learning techniques.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
from typing import List, Dict, Optional, Tuple, Any
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
import warnings
warnings.filterwarnings('ignore')

from config import CONFIG, RISK_THRESHOLDS, VIZ_CONFIG
from utils import *

class DataCollector:
    """Collects and preprocesses financial data for anomaly detection."""
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logging.getLogger(__name__)
        
    def fetch_stock_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch historical stock data for all symbols."""
        stock_data = {}
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=self.start_date, end=self.end_date)
                
                if len(df) == 0:
                    self.logger.warning(f"No data found for {symbol}")
                    continue
                
                # Clean and validate data
                df = self._clean_stock_data(df, symbol)
                stock_data[symbol] = df
                
                self.logger.info(f"Fetched {len(df)} records for {symbol}")
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {e}")
                
        return stock_data
    
    def _clean_stock_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and validate stock data."""
        # Remove rows with missing critical data
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Calculate additional metrics
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Price_MA'] = df['Close'].rolling(window=20).mean()
        
        # Add symbol identifier
        df['Symbol'] = symbol
        
        return df
    
    def get_market_data(self) -> pd.DataFrame:
        """Fetch market benchmark data (S&P 500)."""
        try:
            spy = yf.Ticker("SPY")
            market_data = spy.history(start=self.start_date, end=self.end_date)
            market_data['Market_Returns'] = market_data['Close'].pct_change()
            return market_data[['Close', 'Volume', 'Market_Returns']].rename(
                columns={'Close': 'Market_Close', 'Volume': 'Market_Volume'}
            )
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            return pd.DataFrame()

class FeatureEngineer:
    """Generates features for anomaly detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config['features']
        self.logger = logging.getLogger(__name__)
    
    def create_features(self, stock_data: Dict[str, pd.DataFrame], market_data: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for all symbols."""
        all_features = []
        
        for symbol, df in stock_data.items():
            # Merge with market data
            df_with_market = df.merge(market_data, left_index=True, right_index=True, how='left')
            
            # Generate features
            features = self._generate_symbol_features(df_with_market, symbol)
            all_features.append(features)
        
        # Combine all symbols
        combined_features = pd.concat(all_features, ignore_index=False)
        combined_features = combined_features.sort_index()
        
        self.logger.info(f"Generated {len(combined_features.columns)} features for {len(combined_features)} observations")
        return combined_features
    
    def _generate_symbol_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate features for a single symbol."""
        features = df.copy()
        
        # Ensure we have minimum required data
        if len(features) < 100:
            self.logger.warning(f"Insufficient data for {symbol}: {len(features)} rows")
            return features
        
        # Volume-based features with proper error handling
        try:
            features['Volume_MA'] = df['Volume'].rolling(self.config['volume_window'], min_periods=1).mean()
            features['Volume_Std'] = df['Volume'].rolling(self.config['volume_window'], min_periods=1).std()
            
            # Handle division by zero
            volume_std = features['Volume_Std'].replace(0, 1)
            features['Volume_Z_Score'] = (df['Volume'] - features['Volume_MA']) / volume_std
            features['Volume_Percentile'] = df['Volume'].rolling(self.config['volume_window'], min_periods=1).rank(pct=True)
            
            # Handle division by zero for volume ratio
            volume_ma_safe = features['Volume_MA'].replace(0, 1)
            features['Volume_Ratio'] = df['Volume'] / volume_ma_safe
        except Exception as e:
            self.logger.error(f"Error calculating volume features for {symbol}: {e}")
            features['Volume_MA'] = df['Volume']
            features['Volume_Std'] = 1
            features['Volume_Z_Score'] = 0
            features['Volume_Percentile'] = 0.5
            features['Volume_Ratio'] = 1
        
        # Price-based features with proper error handling
        try:
            features['Volatility'] = df['Returns'].rolling(self.config['volatility_window'], min_periods=1).std()
            features['Price_Momentum'] = df['Close'].pct_change(self.config['momentum_window'])
            features['RSI'] = self._calculate_rsi(df['Close'], self.config['rsi_window'])
            features['Bollinger_Position'] = self._calculate_bollinger_position(df['Close'], self.config['bollinger_window'])
        except Exception as e:
            self.logger.error(f"Error calculating price features for {symbol}: {e}")
            features['Volatility'] = 0.01
            features['Price_Momentum'] = 0
            features['RSI'] = 50
            features['Bollinger_Position'] = 0.5
        
        # Market relative features
        try:
            if 'Market_Returns' in df.columns and not df['Market_Returns'].isna().all():
                market_var = df['Market_Returns'].rolling(60, min_periods=20).var()
                market_var_safe = market_var.replace(0, 0.0001)
                features['Beta'] = df['Returns'].rolling(60, min_periods=20).cov(df['Market_Returns']) / market_var_safe
                features['Alpha'] = df['Returns'] - features['Beta'] * df['Market_Returns']
                
                market_volume_safe = df['Market_Volume'].replace(0, 1)
                features['Relative_Volume'] = df['Volume'] / market_volume_safe
            else:
                features['Beta'] = 1.0
                features['Alpha'] = df['Returns'].fillna(0)
                features['Relative_Volume'] = 1.0
        except Exception as e:
            self.logger.error(f"Error calculating market features for {symbol}: {e}")
            features['Beta'] = 1.0
            features['Alpha'] = 0
            features['Relative_Volume'] = 1.0
        
        # Technical indicators
        try:
            features['MACD'], features['MACD_Signal'] = self._calculate_macd(
                df['Close'], 
                self.config['macd_fast'], 
                self.config['macd_slow'], 
                self.config['macd_signal']
            )
            
            price_ma = df['Close'].rolling(20, min_periods=1).mean()
            features['Price_vs_MA'] = (df['Close'] - price_ma) / price_ma
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators for {symbol}: {e}")
            features['MACD'] = 0
            features['MACD_Signal'] = 0
            features['Price_vs_MA'] = 0
        
        # Anomaly indicators with safe operations
        try:
            features['High_Volume_Flag'] = (features['Volume_Z_Score'].fillna(0) > 2).astype(int)
            vol_quantile = features['Volatility'].rolling(60, min_periods=20).quantile(0.95)
            features['High_Volatility_Flag'] = (features['Volatility'] > vol_quantile).astype(int)
            
            returns_quantile = df['Returns'].abs().rolling(60, min_periods=20).quantile(0.95)
            features['Large_Move_Flag'] = (abs(df['Returns']) > returns_quantile).astype(int)
        except Exception as e:
            self.logger.error(f"Error calculating anomaly flags for {symbol}: {e}")
            features['High_Volume_Flag'] = 0
            features['High_Volatility_Flag'] = 0
            features['Large_Move_Flag'] = 0
        
        # Lag features with safe operations
        try:
            for lag in [1, 2, 3, 5]:
                features[f'Volume_Z_Score_Lag_{lag}'] = features['Volume_Z_Score'].shift(lag)
                features[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        except Exception as e:
            self.logger.error(f"Error calculating lag features for {symbol}: {e}")
            for lag in [1, 2, 3, 5]:
                features[f'Volume_Z_Score_Lag_{lag}'] = 0
                features[f'Returns_Lag_{lag}'] = 0
        
        # Fill any remaining NaN values using forward fill then zero fill
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            features[col] = features[col].fillna(method='ffill').fillna(0)
        
        # Ensure all features have the same length as the original dataframe
        features = features.reindex(df.index, fill_value=0)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_position(self, prices: pd.Series, window: int = 20, std_dev: int = 2) -> pd.Series:
        """Calculate position within Bollinger Bands."""
        ma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return (prices - lower) / (upper - lower)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

class AnomalyDetector:
    """Detects anomalies using multiple methods."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config['detection']
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        
        # Initialize detectors
        self.isolation_forest = IsolationForest(
            contamination=self.config['contamination'],
            random_state=42,
            n_estimators=100
        )
        
        self.lof = LocalOutlierFactor(
            n_neighbors=self.config['n_neighbors'],
            contamination=self.config['contamination']
        )
    
    def detect_anomalies(self, features: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using multiple methods."""
        if features.empty:
            self.logger.warning("Empty features DataFrame provided")
            return features
        
        results = features.copy()
        
        try:
            # Prepare feature matrix - select only numeric columns
            numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
            # Remove non-feature columns
            exclude_cols = ['Symbol'] if 'Symbol' in numeric_cols else []
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if not feature_cols:
                self.logger.warning("No numeric feature columns found")
                return self._add_default_anomaly_columns(results)
            
            # Extract and clean feature matrix
            X = features[feature_cols].copy()
            X = X.fillna(X.median())
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(0)
            
            # Ensure we have enough data
            if len(X) < 10:
                self.logger.warning(f"Insufficient data points: {len(X)}")
                return self._add_default_anomaly_columns(results)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Isolation Forest
            if_predictions = self.isolation_forest.fit_predict(X_scaled)
            if_scores = self.isolation_forest.score_samples(X_scaled)
            
            # Local Outlier Factor
            lof_predictions = self.lof.fit_predict(X_scaled)
            lof_scores = self.lof.negative_outlier_factor_
            
            # Ensure predictions match the original index
            results['IF_Anomaly'] = pd.Series(if_predictions, index=features.index)
            results['IF_Score'] = pd.Series(if_scores, index=features.index)
            results['LOF_Anomaly'] = pd.Series(lof_predictions, index=features.index)
            results['LOF_Score'] = pd.Series(lof_scores, index=features.index)
            
            # Statistical anomalies
            results = self._statistical_anomalies(results)
            
            # Ensemble anomaly score
            results['Ensemble_Score'] = self._calculate_ensemble_score(results)
            results['Ensemble_Anomaly'] = (results['Ensemble_Score'] > self.config['anomaly_threshold']).astype(int)
            
            # Risk assessment
            results['Risk_Level'] = results['Ensemble_Score'].apply(
                lambda x: get_risk_level(x, RISK_THRESHOLDS)
            )
            
            # Summary statistics
            anomaly_rate = (results['Ensemble_Anomaly'] == 1).mean()
            self.logger.info(f"Detected {anomaly_rate:.2%} anomalies in dataset")
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            results = self._add_default_anomaly_columns(results)
        
        return results
    
    def _add_default_anomaly_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add default anomaly columns when detection fails."""
        df['IF_Anomaly'] = 1
        df['IF_Score'] = -0.5
        df['LOF_Anomaly'] = 1
        df['LOF_Score'] = -1.0
        df['Statistical_Score'] = 0.0
        df['Ensemble_Score'] = 1.0
        df['Ensemble_Anomaly'] = 0
        df['Risk_Level'] = 'LOW'
        return df
    
    def _statistical_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect statistical anomalies."""
        # Volume anomalies
        df['Volume_Anomaly'] = detect_outliers_zscore(df['Volume'], threshold=3.0).astype(int)
        
        # Price movement anomalies
        df['Returns_Anomaly'] = detect_outliers_zscore(df['Returns'], threshold=2.5).astype(int)
        
        # Combined statistical score
        df['Statistical_Score'] = (
            abs(df['Volume_Z_Score'].fillna(0)) * 0.4 +
            abs(df['Returns'].fillna(0)) / df['Returns'].std() * 0.3 +
            df['High_Volume_Flag'] * 0.3
        )
        
        return df
    
    def _calculate_ensemble_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ensemble anomaly score."""
        scores = {}
        
        # Normalize IF scores
        if 'IF_Score' in df.columns:
            scores['isolation_forest'] = normalize_scores(-df['IF_Score'])  # Negative because lower is more anomalous
        
        # Normalize LOF scores
        if 'LOF_Score' in df.columns:
            scores['local_outlier_factor'] = normalize_scores(-df['LOF_Score'])  # Negative because lower is more anomalous
        
        # Statistical score
        if 'Statistical_Score' in df.columns:
            scores['statistical'] = normalize_scores(df['Statistical_Score'])
        
        # Aggregate scores
        return aggregate_anomaly_scores(scores, self.config['ensemble_weights'])

class PatternRecognizer:
    """Recognizes specific insider trading patterns."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def identify_patterns(self, features: pd.DataFrame) -> pd.DataFrame:
        """Identify specific insider trading patterns."""
        results = features.copy()
        
        # Pattern 1: Unusual volume before price movement
        results['Pre_Event_Volume_Pattern'] = self._detect_pre_event_volume(results)
        
        # Pattern 2: Consistent insider accumulation
        results['Accumulation_Pattern'] = self._detect_accumulation_pattern(results)
        
        # Pattern 3: Coordinated trading
        results['Coordinated_Pattern'] = self._detect_coordinated_trading(results)
        
        # Pattern 4: News-based anomalies
        results['News_Correlation_Pattern'] = self._detect_news_correlation(results)
        
        # Aggregate pattern score
        pattern_columns = [col for col in results.columns if col.endswith('_Pattern')]
        results['Pattern_Score'] = results[pattern_columns].sum(axis=1)
        
        return results
    
    def _detect_pre_event_volume(self, df: pd.DataFrame) -> pd.Series:
        """Detect unusual volume patterns before significant price movements."""
        pattern_score = pd.Series(0, index=df.index)
        
        # Look for volume spikes 1-5 days before large price movements
        for i in range(5, len(df)):
            current_return = abs(df['Returns'].iloc[i])
            if current_return > df['Returns'].std() * 2:  # Significant price movement
                # Check for volume anomalies in preceding days
                preceding_volume = df['Volume_Z_Score'].iloc[i-5:i]
                if (preceding_volume > 2.0).any():
                    pattern_score.iloc[i] = 1
        
        return pattern_score
    
    def _detect_accumulation_pattern(self, df: pd.DataFrame) -> pd.Series:
        """Detect consistent accumulation patterns."""
        pattern_score = pd.Series(0, index=df.index)
        
        # Look for consistent above-average volume with positive returns
        window = 10
        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i]
            
            volume_condition = (window_data['Volume_Ratio'] > 1.2).sum() >= window * 0.6
            return_condition = window_data['Returns'].sum() > 0
            
            if volume_condition and return_condition:
                pattern_score.iloc[i] = 1
        
        return pattern_score
    
    def _detect_coordinated_trading(self, df: pd.DataFrame) -> pd.Series:
        """Detect coordinated trading patterns across time."""
        pattern_score = pd.Series(0, index=df.index)
        
        # Look for repeated patterns at similar times
        if 'Symbol' in df.columns:
            # This would require cross-symbol analysis
            # For now, detect within-symbol coordination
            volume_pattern = df['Volume_Z_Score'].rolling(5).mean()
            pattern_score = (volume_pattern > 1.5).astype(int)
        
        return pattern_score
    
    def _detect_news_correlation(self, df: pd.DataFrame) -> pd.Series:
        """Detect patterns that correlate with potential news events."""
        pattern_score = pd.Series(0, index=df.index)
        
        # Look for volume/price spikes on specific days of week or times
        # This is a simplified version - real implementation would use news data
        
        # Check for Monday morning patterns (common for news-based trading)
        if hasattr(df.index, 'dayofweek'):
            monday_condition = df.index.dayofweek == 0
            volume_condition = df['Volume_Z_Score'] > 2.0
            pattern_score = (monday_condition & volume_condition).astype(int)
        
        return pattern_score

class EventAnalyzer:
    """Analyzes correlation between trading patterns and corporate events."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config['events']
        self.logger = logging.getLogger(__name__)
    
    def analyze_event_correlation(self, features: pd.DataFrame, events: pd.DataFrame = None) -> Dict[str, Any]:
        """Analyze correlation between anomalies and events."""
        if events is None or events.empty:
            # Generate synthetic events for demonstration
            events = self._generate_synthetic_events(features)
        
        correlations = {}
        
        for symbol in features['Symbol'].unique():
            symbol_data = features[features['Symbol'] == symbol].copy()
            symbol_events = events[events['Symbol'] == symbol] if 'Symbol' in events.columns else events
            
            if len(symbol_events) > 0:
                correlations[symbol] = self._calculate_event_correlations(symbol_data, symbol_events)
        
        return correlations
    
    def _generate_synthetic_events(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic events based on high anomaly scores."""
        events = []
        
        for symbol in features['Symbol'].unique():
            symbol_data = features[features['Symbol'] == symbol]
            
            # Find high anomaly score dates
            high_anomaly_dates = symbol_data[
                symbol_data['Ensemble_Score'] > symbol_data['Ensemble_Score'].quantile(0.9)
            ].index
            
            for date in high_anomaly_dates[:5]:  # Limit to top 5 events per symbol
                events.append({
                    'Symbol': symbol,
                    'Date': date,
                    'Type': 'earnings',  # Simplified
                    'Impact': symbol_data.loc[date, 'Returns'] if date in symbol_data.index else 0
                })
        
        return pd.DataFrame(events)
    
    def _calculate_event_correlations(self, data: pd.DataFrame, events: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlations between events and trading patterns."""
        correlations = {}
        
        for _, event in events.iterrows():
            event_date = pd.to_datetime(event['Date'])
            
            # Find pre-event window
            pre_start = event_date - timedelta(days=self.config['pre_event_window'])
            pre_end = event_date - timedelta(days=1)
            
            # Find post-event window
            post_start = event_date
            post_end = event_date + timedelta(days=self.config['post_event_window'])
            
            pre_data = data[(data.index >= pre_start) & (data.index <= pre_end)]
            post_data = data[(data.index >= post_start) & (data.index <= post_end)]
            
            if len(pre_data) > 0 and len(post_data) > 0:
                # Calculate correlation metrics
                pre_volume_avg = pre_data['Volume_Z_Score'].mean()
                post_return_avg = post_data['Returns'].mean()
                
                correlations[f"{event['Type']}_{event_date.strftime('%Y-%m-%d')}"] = {
                    'pre_volume_anomaly': pre_volume_avg,
                    'post_return': post_return_avg,
                    'correlation': pre_volume_avg * abs(post_return_avg)
                }
        
        return correlations

class MLClassifier:
    """Machine learning classifier for insider trading detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config['ml']
        self.scaler = StandardScaler()
        self.models = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models."""
        self.models['random_forest'] = RandomForestClassifier(
            **self.config['models']['random_forest'],
            random_state=self.config['random_state']
        )
        
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            **self.config['models']['gradient_boosting'],
            random_state=self.config['random_state']
        )
        
        self.models['logistic_regression'] = LogisticRegression(
            **self.config['models']['logistic_regression'],
            random_state=self.config['random_state']
        )
    
    def train_models(self, features: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Train all ML models."""
        # Prepare features and labels
        X, y = self._prepare_training_data(features)
        
        if len(X) == 0 or y.sum() < 10:  # Need at least 10 positive samples
            self.logger.warning("Insufficient training data for ML models")
            return {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config['test_size'], 
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                train_score = model.score(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, 
                    cv=self.config['cv_folds'], scoring='roc_auc'
                )
                
                results[name] = {
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'cv_auc_mean': cv_scores.mean(),
                    'cv_auc_std': cv_scores.std()
                }
                
                self.logger.info(f"{name} - Test Accuracy: {test_score:.3f}, CV AUC: {cv_scores.mean():.3f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")
        
        return results
    
    def _prepare_training_data(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from features."""
        # Select feature columns
        feature_cols = [
            'Volume_Z_Score', 'Volume_Ratio', 'Volatility', 'RSI', 
            'Bollinger_Position', 'MACD', 'Beta', 'Alpha',
            'High_Volume_Flag', 'High_Volatility_Flag', 'Large_Move_Flag'
        ]
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in features.columns]
        
        if not available_cols:
            return pd.DataFrame(), pd.Series()
        
        X = features[available_cols].copy()
        X = clean_numeric_data(X)
        
        # Create labels based on ensemble anomaly detection
        y = features['Ensemble_Anomaly'] if 'Ensemble_Anomaly' in features.columns else pd.Series(0, index=features.index)
        
        # Remove rows with missing data
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        return X, y
    
    def predict_anomalies(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict anomalies using trained models."""
        results = features.copy()
        
        X, _ = self._prepare_training_data(features)
        
        if len(X) == 0:
            return results
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X_scaled)[:, 1]
                    results.loc[X.index, f'{name}_proba'] = probas
                
                predictions = model.predict(X_scaled)
                results.loc[X.index, f'{name}_prediction'] = predictions
                
            except Exception as e:
                self.logger.error(f"Error predicting with {name}: {e}")
        
        return results

class Visualizer:
    """Creates visualizations for insider trading detection results."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.colors = VIZ_CONFIG['colors']
        self.logger = logging.getLogger(__name__)
    
    def create_anomaly_dashboard(self, results: pd.DataFrame, symbol: str = None) -> Dict[str, str]:
        """Create comprehensive anomaly detection dashboard."""
        plots = {}
        
        if symbol:
            data = results[results['Symbol'] == symbol].copy()
            title_prefix = f"{symbol} - "
        else:
            data = results.copy()
            title_prefix = "All Symbols - "
        
        if data.empty:
            return plots
        
        # Time series plot
        plots['timeseries'] = self._create_timeseries_plot(data, title_prefix)
        
        # Anomaly distribution
        plots['distribution'] = self._create_anomaly_distribution(data, title_prefix)
        
        # Correlation heatmap
        plots['correlation'] = self._create_correlation_heatmap(data, title_prefix)
        
        # Risk level distribution
        plots['risk_levels'] = self._create_risk_level_plot(data, title_prefix)
        
        return plots
    
    def _create_timeseries_plot(self, data: pd.DataFrame, title_prefix: str) -> str:
        """Create time series plot with anomalies highlighted."""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Price & Volume', 'Anomaly Scores', 'Risk Levels'],
            specs=[[{"secondary_y": True}], [{}], [{}]],
            vertical_spacing=0.1
        )
        
        # Price and Volume
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Close'], name='Price', line=dict(color=self.colors['primary'])),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=data.index, y=data['Volume'], name='Volume', opacity=0.3, yaxis='y2'),
            row=1, col=1, secondary_y=True
        )
        
        # Highlight anomalies
        anomalies = data[data['Ensemble_Anomaly'] == 1]
        if not anomalies.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomalies.index, y=anomalies['Close'], 
                    mode='markers', name='Anomalies',
                    marker=dict(color=self.colors['danger'], size=8, symbol='triangle-up')
                ),
                row=1, col=1
            )
        
        # Anomaly scores
        fig.add_trace(
            go.Scatter(x=data.index, y=data['Ensemble_Score'], name='Ensemble Score', 
                      line=dict(color=self.colors['warning'])),
            row=2, col=1
        )
        
        # Risk levels
        risk_numeric = data['Risk_Level'].map({
            'NORMAL': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4
        })
        fig.add_trace(
            go.Bar(x=data.index, y=risk_numeric, name='Risk Level',
                  marker=dict(color=risk_numeric, colorscale='Reds')),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f"{title_prefix}Anomaly Detection Timeline",
            height=800,
            showlegend=True
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='timeseries_plot')
    
    def _create_anomaly_distribution(self, data: pd.DataFrame, title_prefix: str) -> str:
        """Create anomaly score distribution plot."""
        fig = go.Figure()
        
        # Histogram of ensemble scores
        fig.add_trace(go.Histogram(
            x=data['Ensemble_Score'],
            nbinsx=50,
            name='Ensemble Score Distribution',
            marker=dict(color=self.colors['primary'], opacity=0.7)
        ))
        
        # Add threshold line
        threshold = self.config.get('detection', {}).get('anomaly_threshold', 2.5)
        fig.add_vline(
            x=threshold, 
            line_dash="dash", 
            line_color=self.colors['danger'],
            annotation_text=f"Threshold: {threshold}"
        )
        
        fig.update_layout(
            title=f"{title_prefix}Anomaly Score Distribution",
            xaxis_title="Ensemble Score",
            yaxis_title="Frequency",
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='distribution_plot')
    
    def _create_correlation_heatmap(self, data: pd.DataFrame, title_prefix: str) -> str:
        """Create correlation heatmap of key features."""
        # Select key numeric columns
        key_cols = [
            'Volume_Z_Score', 'Returns', 'Volatility', 'RSI', 
            'Ensemble_Score', 'IF_Score', 'LOF_Score'
        ]
        available_cols = [col for col in key_cols if col in data.columns]
        
        if len(available_cols) < 2:
            return "<p>Insufficient data for correlation analysis</p>"
        
        corr_matrix = data[available_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title=f"{title_prefix}Feature Correlation Heatmap",
            height=500
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='correlation_plot')
    
    def _create_risk_level_plot(self, data: pd.DataFrame, title_prefix: str) -> str:
        """Create risk level distribution plot."""
        if 'Risk_Level' not in data.columns:
            return "<p>Risk level data not available</p>"
        
        risk_counts = data['Risk_Level'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=risk_counts.index,
                y=risk_counts.values,
                marker=dict(
                    color=['green', 'yellow', 'orange', 'red', 'darkred'][:len(risk_counts)]
                )
            )
        ])
        
        fig.update_layout(
            title=f"{title_prefix}Risk Level Distribution",
            xaxis_title="Risk Level",
            yaxis_title="Count",
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id='risk_levels_plot')

class ReportGenerator:
    """Generates comprehensive reports for insider trading detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_summary_report(self, results: pd.DataFrame, ml_results: Dict[str, Dict[str, float]] = None) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self._generate_summary_stats(results),
            'symbols': self._generate_symbol_reports(results),
            'patterns': self._generate_pattern_analysis(results),
            'ml_performance': ml_results or {},
            'alerts': self._generate_alerts(results)
        }
        
        return report
    
    def _generate_summary_stats(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Generate overall summary statistics."""
        total_observations = len(results)
        total_anomalies = (results['Ensemble_Anomaly'] == 1).sum() if 'Ensemble_Anomaly' in results.columns else 0
        
        risk_distribution = {}
        if 'Risk_Level' in results.columns:
            risk_distribution = results['Risk_Level'].value_counts().to_dict()
        
        avg_ensemble_score = results['Ensemble_Score'].mean() if 'Ensemble_Score' in results.columns else 0
        
        return {
            'total_observations': int(total_observations),
            'total_anomalies': int(total_anomalies),
            'anomaly_rate': float(total_anomalies / total_observations) if total_observations > 0 else 0,
            'average_ensemble_score': float(avg_ensemble_score),
            'risk_distribution': risk_distribution,
            'symbols_analyzed': list(results['Symbol'].unique()) if 'Symbol' in results.columns else []
        }
    
    def _generate_symbol_reports(self, results: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Generate individual symbol reports."""
        symbol_reports = {}
        
        if 'Symbol' not in results.columns:
            return symbol_reports
        
        for symbol in results['Symbol'].unique():
            symbol_data = results[results['Symbol'] == symbol]
            
            symbol_reports[symbol] = {
                'total_observations': len(symbol_data),
                'anomalies': (symbol_data['Ensemble_Anomaly'] == 1).sum() if 'Ensemble_Anomaly' in symbol_data.columns else 0,
                'avg_ensemble_score': symbol_data['Ensemble_Score'].mean() if 'Ensemble_Score' in symbol_data.columns else 0,
                'max_ensemble_score': symbol_data['Ensemble_Score'].max() if 'Ensemble_Score' in symbol_data.columns else 0,
                'high_risk_days': (symbol_data['Risk_Level'].isin(['HIGH', 'CRITICAL'])).sum() if 'Risk_Level' in symbol_data.columns else 0,
                'latest_risk_level': symbol_data['Risk_Level'].iloc[-1] if 'Risk_Level' in symbol_data.columns and len(symbol_data) > 0 else 'UNKNOWN'
            }
        
        return symbol_reports
    
    def _generate_pattern_analysis(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Generate pattern analysis report."""
        pattern_analysis = {}
        
        # Look for pattern columns
        pattern_cols = [col for col in results.columns if col.endswith('_Pattern')]
        
        if pattern_cols:
            for col in pattern_cols:
                pattern_name = col.replace('_Pattern', '')
                pattern_analysis[pattern_name] = {
                    'total_occurrences': int(results[col].sum()),
                    'occurrence_rate': float(results[col].mean())
                }
        
        return pattern_analysis
    
    def _generate_alerts(self, results: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate alerts for high-risk situations."""
        alerts = []
        
        if 'Risk_Level' not in results.columns or 'Symbol' not in results.columns:
            return alerts
        
        # High-risk alerts
        high_risk_data = results[results['Risk_Level'].isin(['HIGH', 'CRITICAL'])]
        
        for idx, row in high_risk_data.iterrows():
            alerts.append({
                'type': 'HIGH_RISK_ANOMALY',
                'symbol': row['Symbol'],
                'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                'risk_level': row['Risk_Level'],
                'ensemble_score': float(row['Ensemble_Score']) if 'Ensemble_Score' in row else 0,
                'message': f"High risk anomaly detected for {row['Symbol']} with score {row.get('Ensemble_Score', 0):.2f}"
            })
        
        # Recent anomaly alerts
        recent_data = results.tail(50)  # Last 50 observations
        recent_anomalies = recent_data[recent_data['Ensemble_Anomaly'] == 1] if 'Ensemble_Anomaly' in recent_data.columns else pd.DataFrame()
        
        for idx, row in recent_anomalies.iterrows():
            alerts.append({
                'type': 'RECENT_ANOMALY',
                'symbol': row['Symbol'],
                'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                'ensemble_score': float(row['Ensemble_Score']) if 'Ensemble_Score' in row else 0,
                'message': f"Recent anomaly detected for {row['Symbol']}"
            })
        
        return alerts

class InsiderTradingDetector:
    """Main class that orchestrates the entire insider trading detection system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or CONFIG
        self.logger = setup_logging(self.config)
        
        # Initialize components
        self.data_collector = None
        self.feature_engineer = FeatureEngineer(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.pattern_recognizer = PatternRecognizer(self.config)
        self.event_analyzer = EventAnalyzer(self.config)
        self.ml_classifier = MLClassifier(self.config)
        self.visualizer = Visualizer(self.config)
        self.report_generator = ReportGenerator(self.config)
        
        # Results storage
        self.results = None
        self.ml_results = None
        self.report = None
    
    def run_analysis(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Run complete insider trading detection analysis."""
        try:
            # Use provided symbols or default from config
            symbols = symbols or self.config['data']['symbols']
            
            # Initialize data collector
            self.data_collector = DataCollector(
                symbols=symbols,
                start_date=self.config['data']['start_date'],
                end_date=self.config['data']['end_date']
            )
            
            self.logger.info("Starting insider trading detection analysis...")
            
            # Step 1: Collect data
            self.logger.info("Collecting stock data...")
            stock_data = self.data_collector.fetch_stock_data()
            
            if not stock_data:
                raise ValueError("No stock data could be collected")
            
            # Step 2: Get market data
            self.logger.info("Collecting market data...")
            market_data = self.data_collector.get_market_data()
            
            # Step 3: Engineer features
            self.logger.info("Engineering features...")
            features = self.feature_engineer.create_features(stock_data, market_data)
            
            # Step 4: Detect anomalies
            self.logger.info("Detecting anomalies...")
            anomaly_results = self.anomaly_detector.detect_anomalies(features)
            
            # Step 5: Recognize patterns
            self.logger.info("Recognizing patterns...")
            pattern_results = self.pattern_recognizer.identify_patterns(anomaly_results)
            
            # Step 6: Train ML models
            self.logger.info("Training ML models...")
            ml_results = self.ml_classifier.train_models(pattern_results)
            
            # Step 7: Apply ML predictions
            self.logger.info("Applying ML predictions...")
            ml_predictions = self.ml_classifier.predict_anomalies(pattern_results)
            
            # Step 8: Analyze events
            self.logger.info("Analyzing event correlations...")
            event_correlations = self.event_analyzer.analyze_event_correlation(ml_predictions)
            
            # Store results
            self.results = ml_predictions
            self.ml_results = ml_results
            
            # Step 9: Generate report
            self.logger.info("Generating report...")
            self.report = self.report_generator.generate_summary_report(
                self.results, self.ml_results
            )
            
            self.logger.info("Analysis completed successfully")
            
            return {
                'success': True,
                'results': self.results,
                'ml_results': self.ml_results,
                'report': self.report,
                'event_correlations': event_correlations
            }
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': None,
                'report': None
            }
    
    def get_visualizations(self, symbol: str = None) -> Dict[str, str]:
        """Get visualizations for the analysis results."""
        if self.results is None:
            return {'error': 'No analysis results available. Run analysis first.'}
        
        return self.visualizer.create_anomaly_dashboard(self.results, symbol)
    
    def get_symbol_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get detailed analysis for a specific symbol."""
        if self.results is None:
            return {'error': 'No analysis results available. Run analysis first.'}
        
        symbol_data = self.results[self.results['Symbol'] == symbol]
        
        if symbol_data.empty:
            return {'error': f'No data available for symbol {symbol}'}
        
        return {
            'symbol': symbol,
            'data': symbol_data.to_dict('records'),
            'summary': {
                'total_observations': len(symbol_data),
                'anomalies': int((symbol_data['Ensemble_Anomaly'] == 1).sum()) if 'Ensemble_Anomaly' in symbol_data.columns else 0,
                'avg_score': float(symbol_data['Ensemble_Score'].mean()) if 'Ensemble_Score' in symbol_data.columns else 0,
                'max_score': float(symbol_data['Ensemble_Score'].max()) if 'Ensemble_Score' in symbol_data.columns else 0,
                'current_risk': symbol_data['Risk_Level'].iloc[-1] if 'Risk_Level' in symbol_data.columns and len(symbol_data) > 0 else 'UNKNOWN'
            },
            'visualizations': self.get_visualizations(symbol)
        }
