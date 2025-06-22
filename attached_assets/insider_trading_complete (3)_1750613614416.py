#!/usr/bin/env python3
"""
Insider Trading Signal Detection System
A comprehensive system for detecting potential insider trading patterns
using anomaly detection and machine learning techniques.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time
from typing import List, Dict, Optional, Tuple
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('insider_trading_detection.log'),
        logging.StreamHandler()
    ]
)

# Configuration
CONFIG = {
    'data': {
        'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
        'start_date': '2022-01-01',
        'end_date': '2024-01-01'
    },
    'features': {
        'volume_window': 20,
        'volatility_window': 30,
        'momentum_window': 10
    },
    'detection': {
        'contamination': 0.05,
        'n_neighbors': 20,
        'anomaly_threshold': 2.5
    },
    'events': {
        'pre_event_window': 30,
        'post_event_window': 5
    },
    'ml': {
        'test_size': 0.3,
        'cv_folds': 5,
        'random_state': 42
    }
}

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
    
    def simulate_events(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Simulate corporate events for testing purposes."""
        events = []
        np.random.seed(42)  # For reproducible results
        
        for symbol, df in stock_data.items():
            # Generate events based on high volatility days
            high_vol_days = df[df['Returns'].abs() > df['Returns'].std() * 2].copy()
            
            if len(high_vol_days) == 0:
                continue
                
            n_events = min(len(high_vol_days) // 10, 15)  # Reasonable number of events
            if n_events == 0:
                continue
                
            selected_events = high_vol_days.sample(n=n_events, random_state=42)
            
            for date, row in selected_events.iterrows():
                event_type = np.random.choice(['earnings', 'acquisition', 'regulatory', 'management_change'])
                events.append({
                    'Symbol': symbol,
                    'Date': date,
                    'Type': event_type,
                    'Impact': abs(row['Returns']),
                    'Volume_Spike': row['Volume'] / row['Volume_MA'] if row['Volume_MA'] > 0 else 1
                })
        
        return pd.DataFrame(events)
    
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
    
    def __init__(self, volume_window: int = 20, volatility_window: int = 30, momentum_window: int = 10):
        self.volume_window = volume_window
        self.volatility_window = volatility_window
        self.momentum_window = momentum_window
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
        
        # Volume-based features
        features['Volume_MA'] = df['Volume'].rolling(self.volume_window).mean()
        features['Volume_Std'] = df['Volume'].rolling(self.volume_window).std()
        features['Volume_Z_Score'] = (df['Volume'] - features['Volume_MA']) / features['Volume_Std']
        features['Volume_Percentile'] = df['Volume'].rolling(self.volume_window).rank(pct=True)
        features['Volume_Ratio'] = df['Volume'] / features['Volume_MA']
        
        # Price-based features
        features['Volatility'] = df['Returns'].rolling(self.volatility_window).std()
        features['Price_Momentum'] = df['Close'].pct_change(self.momentum_window)
        features['RSI'] = self._calculate_rsi(df['Close'])
        features['Bollinger_Position'] = self._calculate_bollinger_position(df['Close'])
        
        # Market relative features
        if 'Market_Returns' in df.columns and not df['Market_Returns'].isna().all():
            features['Beta'] = df['Returns'].rolling(60).cov(df['Market_Returns']) / df['Market_Returns'].rolling(60).var()
            features['Alpha'] = df['Returns'] - features['Beta'] * df['Market_Returns']
            features['Relative_Volume'] = df['Volume'] / df['Market_Volume']
        else:
            features['Beta'] = 1.0
            features['Alpha'] = df['Returns']
            features['Relative_Volume'] = 1.0
        
        # Technical indicators
        features['MACD'], features['MACD_Signal'] = self._calculate_macd(df['Close'])
        features['Price_vs_MA'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()
        
        # Anomaly indicators
        features['High_Volume_Flag'] = (features['Volume_Z_Score'] > 2).astype(int)
        features['High_Volatility_Flag'] = (features['Volatility'] > features['Volatility'].rolling(60).quantile(0.95)).astype(int)
        features['Large_Move_Flag'] = (abs(df['Returns']) > df['Returns'].rolling(60).quantile(0.95)).astype(int)
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            features[f'Volume_Z_Score_Lag_{lag}'] = features['Volume_Z_Score'].shift(lag)
            features[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
        
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
    
    def __init__(self, contamination: float = 0.05, n_neighbors: int = 20, threshold: float = 2.5):
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        
        # Initialize detectors
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination
        )
    
    def detect_anomalies(self, features: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using multiple methods."""
        results = features.copy()
        
        # Prepare feature matrix (remove non-numeric columns)
        feature_cols = features.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in feature_cols if col not in ['Symbol']]
        
        X = features[feature_cols].fillna(features[feature_cols].median())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Isolation Forest
        results['IF_Anomaly'] = self.isolation_forest.fit_predict(X_scaled)
        results['IF_Score'] = self.isolation_forest.score_samples(X_scaled)
        
        # Local Outlier Factor
        results['LOF_Anomaly'] = self.lof.fit_predict(X_scaled)
        results['LOF_Score'] = self.lof.negative_outlier_factor_
        
        # Statistical anomalies
        results = self._statistical_anomalies(results)
        
        # Ensemble anomaly score
        results['Ensemble_Score'] = self._calculate_ensemble_score(results)
        results['Ensemble_Anomaly'] = (results['Ensemble_Score'] > self.threshold).astype(int)
        
        # Summary statistics
        anomaly_rate = (results['Ensemble_Anomaly'] == 1).mean()
        self.logger.info(f"Detected {anomaly_rate:.2%} anomalies")
        
        return results
    
    def _statistical_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect statistical anomalies using z-scores and percentiles."""
        # Volume anomalies
        df['Volume_Anomaly'] = (abs(df['Volume_Z_Score']) > self.threshold).astype(int)
        
        # Return anomalies
        if 'Returns' in df.columns:
            return_z = np.abs(stats.zscore(df['Returns'].dropna()))
            df.loc[df['Returns'].notna(), 'Return_Anomaly'] = (return_z > self.threshold).astype(int)
        else:
            df['Return_Anomaly'] = 0
        
        # Combined statistical anomaly
        df['Statistical_Anomaly'] = (
            (df['Volume_Anomaly'] == 1) | 
            (df['Return_Anomaly'] == 1)
        ).astype(int)
        
        return df
    
    def _calculate_ensemble_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ensemble anomaly score."""
        scores = []
        
        # Normalize scores
        if 'IF_Score' in df.columns:
            if_normalized = (df['IF_Score'] - df['IF_Score'].min()) / (df['IF_Score'].max() - df['IF_Score'].min())
            scores.append(1 - if_normalized)  # Lower IF score = more anomalous
        
        if 'LOF_Score' in df.columns:
            lof_normalized = (df['LOF_Score'] - df['LOF_Score'].min()) / (df['LOF_Score'].max() - df['LOF_Score'].min())
            scores.append(1 - lof_normalized)  # Lower LOF score = more anomalous
        
        # Volume and return z-scores
        if 'Volume_Z_Score' in df.columns:
            scores.append(abs(df['Volume_Z_Score']) / 5)  # Normalize to 0-1 range
        
        if 'Returns' in df.columns:
            return_z = abs(stats.zscore(df['Returns'].fillna(0)))
            scores.append(return_z / 5)
        
        # Average ensemble score
        if scores:
            ensemble_score = np.mean(scores, axis=0)
        else:
            ensemble_score = pd.Series(0, index=df.index)
        
        return ensemble_score

class EventAnalyzer:
    """Analyzes correlation between anomalies and corporate events."""
    
    def __init__(self, pre_window: int = 30, post_window: int = 5):
        self.pre_window = pre_window
        self.post_window = post_window
        self.logger = logging.getLogger(__name__)
    
    def analyze_event_correlation(self, anomalies: pd.DataFrame, events: pd.DataFrame) -> Dict:
        """Analyze correlation between detected anomalies and events."""
        results = {
            'event_analysis': [],
            'performance_metrics': {},
            'summary_stats': {}
        }
        
        for _, event in events.iterrows():
            event_results = self._analyze_single_event(anomalies, event)
            results['event_analysis'].append(event_results)
        
        # Calculate overall performance metrics
        results['performance_metrics'] = self._calculate_performance_metrics(results['event_analysis'])
        results['summary_stats'] = self._calculate_summary_stats(results['event_analysis'])
        
        return results
    
    def _analyze_single_event(self, anomalies: pd.DataFrame, event: pd.Series) -> Dict:
        """Analyze anomalies around a single event."""
        symbol = event['Symbol']
        event_date = pd.to_datetime(event['Date'])
        
        # Filter data for the specific symbol
        symbol_data = anomalies[anomalies['Symbol'] == symbol].copy()
        
        # Define time windows
        pre_start = event_date - timedelta(days=self.pre_window)
        pre_end = event_date - timedelta(days=1)
        post_start = event_date
        post_end = event_date + timedelta(days=self.post_window)
        
        # Get pre-event anomalies
        pre_event_data = symbol_data[
            (symbol_data.index >= pre_start) & 
            (symbol_data.index <= pre_end)
        ]
        
        # Get post-event data for validation
        post_event_data = symbol_data[
            (symbol_data.index >= post_start) & 
            (symbol_data.index <= post_end)
        ]
        
        # Analyze anomalies
        pre_anomalies = pre_event_data[pre_event_data['Ensemble_Anomaly'] == 1]
        
        result = {
            'event_id': f"{symbol}_{event_date.strftime('%Y%m%d')}",
            'symbol': symbol,
            'event_date': event_date,
            'event_type': event['Type'],
            'event_impact': event['Impact'],
            'pre_anomaly_count': len(pre_anomalies),
            'pre_anomaly_days': [d.strftime('%Y-%m-%d') for d in pre_anomalies.index],
            'max_pre_anomaly_score': pre_anomalies['Ensemble_Score'].max() if len(pre_anomalies) > 0 else 0,
            'days_before_event': (event_date - pre_anomalies.index.max()).days if len(pre_anomalies) > 0 else None,
            'volume_spike_detected': any(pre_anomalies['Volume_Anomaly'] == 1) if len(pre_anomalies) > 0 else False,
            'price_anomaly_detected': any(pre_anomalies['Return_Anomaly'] == 1) if len(pre_anomalies) > 0 else False,
            'post_event_return': post_event_data['Returns'].sum() if len(post_event_data) > 0 else 0
        }
        
        return result
    
    def _calculate_performance_metrics(self, event_analysis: List[Dict]) -> Dict:
        """Calculate detection performance metrics."""
        if not event_analysis:
            return {}
        
        # Basic detection metrics
        total_events = len(event_analysis)
        detected_events = sum(1 for e in event_analysis if e['pre_anomaly_count'] > 0)
        
        detection_rate = detected_events / total_events if total_events > 0 else 0
        
        # Lead time analysis
        lead_times = [e['days_before_event'] for e in event_analysis if e['days_before_event'] is not None]
        avg_lead_time = np.mean(lead_times) if lead_times else 0
        
        # Impact correlation
        high_impact_events = [e for e in event_analysis if e['event_impact'] > 0.05]  # >5% move
        high_impact_detected = sum(1 for e in high_impact_events if e['pre_anomaly_count'] > 0)
        high_impact_rate = high_impact_detected / len(high_impact_events) if high_impact_events else 0
        
        return {
            'total_events': total_events,
            'detected_events': detected_events,
            'detection_rate': detection_rate,
            'average_lead_time_days': avg_lead_time,
            'high_impact_detection_rate': high_impact_rate,
            'volume_detection_rate': sum(1 for e in event_analysis if e['volume_spike_detected']) / total_events,
            'price_detection_rate': sum(1 for e in event_analysis if e['price_anomaly_detected']) / total_events
        }
    
    def _calculate_summary_stats(self, event_analysis: List[Dict]) -> Dict:
        """Calculate summary statistics."""
        if not event_analysis:
            return {}
        
        anomaly_counts = [e['pre_anomaly_count'] for e in event_analysis]
        anomaly_scores = [e['max_pre_anomaly_score'] for e in event_analysis if e['max_pre_anomaly_score'] > 0]
        
        event_types = [e['event_type'] for e in event_analysis]
        event_type_counts = pd.Series(event_types).value_counts().to_dict()
        
        return {
            'avg_anomalies_per_event': np.mean(anomaly_counts),
            'max_anomalies_per_event': max(anomaly_counts) if anomaly_counts else 0,
            'avg_anomaly_score': np.mean(anomaly_scores) if anomaly_scores else 0,
            'event_type_breakdown': event_type_counts
        }

class MLClassifier:
    """Machine learning layer for anomaly classification."""
    
    def __init__(self, test_size: float = 0.3, cv_folds: int = 5, random_state: int = 42):
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = {}
        self.logger = logging.getLogger(__name__)
    
    def prepare_training_data(self, features: pd.DataFrame, events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data with labels based on events."""
        # Create labels based on proximity to events
        labels = self._create_event_labels(features, events)
        
        # Select relevant features for ML
        feature_cols = self._select_ml_features(features)
        X = features[feature_cols].fillna(features[feature_cols].median())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        return X, labels
    
    def _create_event_labels(self, features: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
        """Create binary labels based on proximity to events."""
        labels = pd.Series(0, index=features.index, name='event_label')
        
        for _, event in events.iterrows():
            symbol = event['Symbol']
            event_date = pd.to_datetime(event['Date'])
            
            # Label pre-event window as positive
            pre_start = event_date - pd.Timedelta(days=30)
            pre_end = event_date - pd.Timedelta(days=1)
            
            # Find matching rows
            mask = (
                (features['Symbol'] == symbol) &
                (features.index >= pre_start) &
                (features.index <= pre_end)
            )
            
            labels.loc[mask] = 1
        
        return labels
    
    def _select_ml_features(self, features: pd.DataFrame) -> List[str]:
        """Select relevant features for ML models."""
        # Exclude non-predictive columns
        exclude_cols = ['Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        
        # Select numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        ml_features = [col for col in numeric_cols if col not in exclude_cols]
        
        return ml_features
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train multiple ML models."""
        if len(X) == 0 or y.sum() == 0:
            self.logger.warning("Insufficient data for ML training")
            return {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, 
            stratify=y if y.sum() > 1 else None
        )
        
        # Scale features for logistic regression
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        try:
            # Logistic Regression
            self.models['logistic'] = LogisticRegression(random_state=self.random_state, max_iter=1000)
            self.models['logistic'].fit(X_train_scaled, y_train)
            results['logistic'] = self._evaluate_model('logistic', X_test_scaled, y_test)
            
            # Random Forest
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )
            self.models['random_forest'].fit(X_train, y_train)
            results['random_forest'] = self._evaluate_model('random_forest', X_test, y_test)
            
            # Feature importance for Random Forest
            if hasattr(self.models['random_forest'], 'feature_importances_'):
                self.feature_importance['random_forest'] = dict(zip(
                    X.columns, self.models['random_forest'].feature_importances_
                ))
        
        except Exception as e:
            self.logger.error(f"Error training ML models: {e}")
        
        return results
    
    def _evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: pd.Series) -> Dict:
        """Evaluate a single model."""
        model = self.models[model_name]
        
        try:
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            results = {
                'accuracy': model.score(X_test, y_test),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            if y_pred_proba is not None:
                results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_name}: {e}")
            return {}

class ComplianceReporter:
    """Generates compliance reports and flags for review."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_daily_report(self, anomalies: pd.DataFrame, date: str) -> Dict:
        """Generate daily compliance report."""
        report_date = pd.to_datetime(date)
        daily_data = anomalies[anomalies.index.date == report_date.date()]
        
        # Identify high-priority flags
        high_priority = daily_data[
            (daily_data['Ensemble_Score'] > 3.0) |
            (daily_data['Volume_Z_Score'].abs() > 3.0)
        ]
        
        report = {
            'report_date': date,
            'total_observations': len(daily_data),
            'total_anomalies': len(daily_data[daily_data['Ensemble_Anomaly'] == 1]),
            'high_priority_flags': len(high_priority),
            'symbols_flagged': high_priority['Symbol'].unique().tolist(),
            'flag_details': []
        }
        
        # Detailed flag information
        for _, row in high_priority.iterrows():
            flag_detail = {
                'symbol': row['Symbol'],
                'timestamp': row.name.isoformat(),
                'ensemble_score': float(row['Ensemble_Score']),
                'volume_z_score': float(row['Volume_Z_Score']) if pd.notna(row['Volume_Z_Score']) else None,
                'price_return': float(row['Returns']) if pd.notna(row['Returns']) else None,
                'recommended_action': self._determine_action(row)
            }
            report['flag_details'].append(flag_detail)
        
        return report
    
    def _determine_action(self, row: pd.Series) -> str:
        """Determine recommended compliance action."""
        score = row['Ensemble_Score']
        volume_z = abs(row.get('Volume_Z_Score', 0))
        
        if score > 4.0 or volume_z > 4.0:
            return "IMMEDIATE_REVIEW"
        elif score > 3.0 or volume_z > 3.0:
            return "PRIORITY_REVIEW"
        else:
            return "STANDARD_REVIEW"
    
    def generate_summary_report(self, anomalies: pd.DataFrame, start_date: str, end_date: str) -> Dict:
        """Generate comprehensive summary report."""
        date_range = pd.date_range(start=start_date, end=end_date)
        
        summary = {
            'report_period': f"{start_date} to {end_date}",
            'total_trading_days': len(date_range),
            'total_observations': len(anomalies),
            'total_anomalies': len(anomalies[anomalies['Ensemble_Anomaly'] == 1]),
            'anomaly_rate': len(anomalies[anomalies['Ensemble_Anomaly'] == 1]) / len(anomalies) if len(anomalies) > 0 else 0,
            'symbols_monitored': anomalies['Symbol'].nunique(),
            'high_priority_days': len(anomalies[anomalies['Ensemble_Score'] > 3.0]),
            'avg_anomaly_score': anomalies['Ensemble_Score'].mean(),
            'max_anomaly_score': anomalies['Ensemble_Score'].max(),
            'compliance_flags': len(anomalies[anomalies['Ensemble_Score'] > 4.0])
        }
        
        return summary


def main():
    """Main execution function for testing."""
    print("Insider Trading Detection System - Test Run")
    print("=" * 50)
    
    # Test with sample symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    # Initialize components
    print("Initializing data collector...")
    collector = DataCollector(symbols, CONFIG['data']['start_date'], CONFIG['data']['end_date'])
    
    print("Fetching stock data...")
    stock_data = collector.fetch_stock_data()
    
    if not stock_data:
        print("No stock data retrieved. Exiting.")
        return
    
    print("Fetching market data...")
    market_data = collector.get_market_data()
    
    print("Generating events...")
    events = collector.simulate_events(stock_data)
    
    print("Engineering features...")
    feature_engineer = FeatureEngineer()
    features = feature_engineer.create_features(stock_data, market_data)
    
    print("Detecting anomalies...")
    detector = AnomalyDetector()
    anomalies = detector.detect_anomalies(features)
    
    print("Analyzing events...")
    event_analyzer = EventAnalyzer()
    event_analysis = event_analyzer.analyze_event_correlation(anomalies, events)
    
    print("Training ML models...")
    ml_classifier = MLClassifier()
    X, y = ml_classifier.prepare_training_data(features, events)
    ml_results = ml_classifier.train_models(X, y)
    
    print("Generating compliance report...")
    reporter = ComplianceReporter()
    latest_date = anomalies.index.max().strftime('%Y-%m-%d')
    daily_report = reporter.generate_daily_report(anomalies, latest_date)
    
    summary_report = reporter.generate_summary_report(
        anomalies, 
        CONFIG['data']['start_date'], 
        CONFIG['data']['end_date']
    )
    
    # Print results
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    
    print(f"Total observations: {len(anomalies):,}")
    print(f"Anomalies detected: {summary_report['total_anomalies']:,}")
    print(f"Anomaly rate: {summary_report['anomaly_rate']:.2%}")
    print(f"High priority flags: {daily_report['high_priority_flags']}")
    
    print("\nEvent Analysis:")
    print(f"Total events: {event_analysis['performance_metrics'].get('total_events', 0)}")
    print(f"Detection rate: {event_analysis['performance_metrics'].get('detection_rate', 0):.2%}")
    print(f"Average lead time: {event_analysis['performance_metrics'].get('average_lead_time_days', 0):.1f} days")
    
    if ml_results:
        print("\nML Model Performance:")
        for model_name, metrics in ml_results.items():
            print(f"{model_name}: {metrics.get('accuracy', 0):.3f} accuracy")
    
    print("\nTop anomalies by score:")
    top_anomalies = anomalies.nlargest(5, 'Ensemble_Score')[['Symbol', 'Ensemble_Score', 'Volume_Z_Score']]
    for _, row in top_anomalies.iterrows():
        print(f"{row['Symbol']}: Score={row['Ensemble_Score']:.3f}, Volume Z={row['Volume_Z_Score']:.2f}")
    
    print("\nAnalysis complete!")
    return anomalies, event_analysis, ml_results, summary_report


if __name__ == "__main__":
    main()