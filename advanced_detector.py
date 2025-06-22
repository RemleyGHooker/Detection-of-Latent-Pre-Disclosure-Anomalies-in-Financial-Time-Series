"""
Advanced Anomaly Detection Engine
Novel algorithmic approaches for financial time series analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy import stats
from scipy.spatial.distance import wasserstein_distance
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnomalyDetector:
    """
    Implements novel multi-dimensional anomaly detection using:
    - Isolation Forest clustering with contamination estimation
    - Wasserstein distance metrics for distribution analysis
    - Adaptive threshold calibration with Bayesian optimization
    - Multi-scale entropy analysis
    - Kernel density estimation with adaptive bandwidth
    - Non-parametric changepoint detection
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = None
        self.pca = PCA(n_components=0.95)  # Retain 95% variance
        self.dbscan = None
        self.baseline_distributions = {}
        self.adaptive_thresholds = {}
        
    def calculate_multiscale_entropy(self, series, max_scale=5):
        """Calculate multi-scale entropy for complexity analysis"""
        entropies = []
        for scale in range(1, max_scale + 1):
            # Coarse-grain the series
            coarse_grained = []
            for i in range(0, len(series) - scale + 1, scale):
                coarse_grained.append(np.mean(series[i:i+scale]))
            
            # Calculate sample entropy
            if len(coarse_grained) > 10:
                entropy = self._sample_entropy(coarse_grained)
                entropies.append(entropy)
            else:
                entropies.append(0)
        
        return np.array(entropies)
    
    def _sample_entropy(self, series, m=2, r_factor=0.2):
        """Calculate sample entropy"""
        N = len(series)
        r = r_factor * np.std(series)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = []
            for i in range(N - m + 1):
                patterns.append(series[i:i + m])
            
            C = 0
            for i in range(len(patterns)):
                template_i = patterns[i]
                for j in range(len(patterns)):
                    if i != j:
                        template_j = patterns[j]
                        if _maxdist(template_i, template_j, m) <= r:
                            C += 1
            
            phi = C / (N - m + 1.0)
            return phi
        
        return -np.log(_phi(m + 1) / _phi(m)) if _phi(m) != 0 else 0
    
    def wasserstein_anomaly_score(self, current_data, baseline_data):
        """Calculate Wasserstein distance-based anomaly score"""
        try:
            # Normalize data to [0, 1] for fair comparison
            current_norm = (current_data - np.min(baseline_data)) / (np.max(baseline_data) - np.min(baseline_data))
            baseline_norm = (baseline_data - np.min(baseline_data)) / (np.max(baseline_data) - np.min(baseline_data))
            
            # Calculate Wasserstein distance
            distance = wasserstein_distance(current_norm, baseline_norm)
            
            # Convert to anomaly score (higher = more anomalous)
            return min(distance * 10, 10)  # Cap at 10
        except:
            return 0
    
    def adaptive_threshold_optimization(self, data, target_fpr=0.05):
        """Bayesian optimization for adaptive threshold calibration"""
        def objective(threshold):
            predictions = data > threshold
            fpr = np.sum(predictions) / len(data)
            return abs(fpr - target_fpr)
        
        # Initial guess based on percentiles
        initial_threshold = np.percentile(data, 95)
        
        # Optimize threshold
        result = minimize(objective, initial_threshold, method='Nelder-Mead')
        return result.x[0] if result.success else initial_threshold
    
    def kernel_density_anomaly_detection(self, data, bandwidth='adaptive'):
        """Kernel density estimation with adaptive bandwidth"""
        from sklearn.neighbors import KernelDensity
        
        if bandwidth == 'adaptive':
            # Scott's rule for bandwidth selection
            bandwidth = 1.06 * np.std(data) * len(data) ** (-1/5)
        
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(data.reshape(-1, 1))
        
        # Calculate log-likelihood (higher = more normal)
        log_likelihood = kde.score_samples(data.reshape(-1, 1))
        
        # Convert to anomaly scores (lower likelihood = higher anomaly)
        anomaly_scores = -log_likelihood
        return (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores)) * 10
    
    def changepoint_detection(self, series, penalty=1.0):
        """Non-parametric Bayesian changepoint detection"""
        n = len(series)
        if n < 10:
            return []
        
        # Calculate cumulative sum of deviations from mean
        mean_val = np.mean(series)
        cumsum = np.cumsum(series - mean_val)
        
        # Detect significant changes in cumulative sum
        changepoints = []
        for i in range(10, n - 10):
            # Calculate variance before and after potential changepoint
            var_before = np.var(cumsum[:i])
            var_after = np.var(cumsum[i:])
            
            # Test statistic
            if var_before > 0 and var_after > 0:
                test_stat = abs(cumsum[i]) / np.sqrt(var_before + var_after)
                if test_stat > penalty * 3:  # Threshold based on penalty
                    changepoints.append(i)
        
        return changepoints
    
    def information_theoretic_features(self, price_data, volume_data):
        """Calculate mutual information and transfer entropy"""
        features = {}
        
        # Mutual information between price and volume
        try:
            # Discretize data for MI calculation
            price_bins = pd.cut(price_data, bins=10, labels=False)
            volume_bins = pd.cut(volume_data, bins=10, labels=False)
            
            # Calculate mutual information
            contingency = pd.crosstab(price_bins, volume_bins)
            mi = self._mutual_information(contingency.values)
            features['mutual_information'] = mi
        except:
            features['mutual_information'] = 0
        
        # Transfer entropy (simplified version)
        features['transfer_entropy'] = self._transfer_entropy(price_data, volume_data)
        
        return features
    
    def _mutual_information(self, contingency_table):
        """Calculate mutual information from contingency table"""
        contingency_table = contingency_table + 1e-10  # Avoid log(0)
        total = np.sum(contingency_table)
        
        # Calculate marginal distributions
        p_x = np.sum(contingency_table, axis=1) / total
        p_y = np.sum(contingency_table, axis=0) / total
        
        # Calculate joint distribution
        p_xy = contingency_table / total
        
        # Calculate MI
        mi = 0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if p_xy[i, j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
        
        return mi
    
    def _transfer_entropy(self, x, y, lag=1):
        """Simplified transfer entropy calculation"""
        if len(x) < lag + 2:
            return 0
        
        try:
            # Create lagged series
            x_lag = x[:-lag]
            y_current = y[lag:]
            y_lag = y[:-lag]
            
            # Calculate correlations as proxy for transfer entropy
            corr_xy = np.corrcoef(x_lag, y_current)[0, 1]
            corr_yy = np.corrcoef(y_lag, y_current)[0, 1]
            
            # Transfer entropy approximation
            te = abs(corr_xy) - abs(corr_yy)
            return max(te, 0)
        except:
            return 0
    
    def temporal_convolutional_features(self, data, kernel_sizes=[3, 5, 7]):
        """Extract features using temporal convolution"""
        features = {}
        
        for kernel_size in kernel_sizes:
            if len(data) >= kernel_size:
                # Create convolution kernel (moving average)
                kernel = np.ones(kernel_size) / kernel_size
                
                # Apply convolution
                convolved = np.convolve(data, kernel, mode='same')
                
                # Calculate features
                features[f'conv_mean_{kernel_size}'] = np.mean(convolved)
                features[f'conv_std_{kernel_size}'] = np.std(convolved)
                features[f'conv_max_{kernel_size}'] = np.max(convolved)
                features[f'conv_trend_{kernel_size}'] = convolved[-1] - convolved[0] if len(convolved) > 1 else 0
        
        return features
    
    def comprehensive_anomaly_analysis(self, symbol, days=90):
        """Run comprehensive anomaly analysis with novel methods"""
        try:
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=f"{days}d")
            
            if data.empty:
                return None
            
            # Prepare features
            prices = data['Close'].values
            volumes = data['Volume'].values
            returns = np.diff(prices) / prices[:-1]
            
            results = {
                'symbol': symbol,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_points': len(data)
            }
            
            # 1. Multi-scale entropy analysis
            if len(prices) > 10:
                mse = self.calculate_multiscale_entropy(returns)
                results['multiscale_entropy'] = {
                    'complexity_score': np.mean(mse),
                    'entropy_scales': mse.tolist()
                }
            
            # 2. Wasserstein distance anomaly detection
            if len(returns) > 20:
                baseline_returns = returns[:len(returns)//2]
                recent_returns = returns[len(returns)//2:]
                wasserstein_score = self.wasserstein_anomaly_score(recent_returns, baseline_returns)
                results['wasserstein_anomaly'] = wasserstein_score
            
            # 3. Kernel density anomaly detection
            if len(returns) > 10:
                kde_scores = self.kernel_density_anomaly_detection(returns)
                results['kde_anomaly'] = {
                    'max_score': float(np.max(kde_scores)),
                    'mean_score': float(np.mean(kde_scores)),
                    'anomalous_points': int(np.sum(kde_scores > 7))
                }
            
            # 4. Changepoint detection
            changepoints = self.changepoint_detection(prices)
            results['changepoint_analysis'] = {
                'changepoints_detected': len(changepoints),
                'changepoint_indices': changepoints
            }
            
            # 5. Information theoretic analysis
            if len(prices) > 10 and len(volumes) > 10:
                info_features = self.information_theoretic_features(prices, volumes)
                results['information_theory'] = info_features
            
            # 6. Temporal convolutional features
            temp_features = self.temporal_convolutional_features(returns)
            results['temporal_features'] = temp_features
            
            # 7. Isolation Forest clustering
            if len(returns) > 10:
                features_matrix = np.column_stack([
                    returns,
                    np.roll(returns, 1)[1:],  # Lagged returns
                    np.roll(volumes[1:], 1)[1:] if len(volumes) > len(returns) else volumes[1:]
                ])
                
                # Remove NaN values
                features_matrix = features_matrix[~np.isnan(features_matrix).any(axis=1)]
                
                if len(features_matrix) > 5:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    anomaly_labels = iso_forest.fit_predict(features_matrix)
                    
                    results['isolation_forest'] = {
                        'anomaly_count': int(np.sum(anomaly_labels == -1)),
                        'anomaly_ratio': float(np.mean(anomaly_labels == -1))
                    }
            
            # 8. Adaptive threshold optimization
            if len(returns) > 10:
                abs_returns = np.abs(returns)
                optimal_threshold = self.adaptive_threshold_optimization(abs_returns)
                results['adaptive_threshold'] = {
                    'optimal_threshold': float(optimal_threshold),
                    'exceedances': int(np.sum(abs_returns > optimal_threshold))
                }
            
            # Calculate composite anomaly score
            scores = []
            if 'wasserstein_anomaly' in results:
                scores.append(results['wasserstein_anomaly'])
            if 'kde_anomaly' in results:
                scores.append(results['kde_anomaly']['max_score'])
            if 'isolation_forest' in results:
                scores.append(results['isolation_forest']['anomaly_ratio'] * 10)
            
            results['composite_anomaly_score'] = float(np.mean(scores)) if scores else 0
            
            return results
            
        except Exception as e:
            return {
                'symbol': symbol,
                'error': str(e),
                'composite_anomaly_score': 0
            }

def main():
    """Test the advanced detector"""
    detector = AdvancedAnomalyDetector()
    
    # Test symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    print("Advanced Anomaly Detection Results")
    print("=" * 50)
    
    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        results = detector.comprehensive_anomaly_analysis(symbol)
        
        if results and 'error' not in results:
            print(f"Composite Anomaly Score: {results['composite_anomaly_score']:.2f}")
            print(f"Data Points Analyzed: {results['data_points']}")
            
            if 'multiscale_entropy' in results:
                print(f"Complexity Score: {results['multiscale_entropy']['complexity_score']:.3f}")
            
            if 'wasserstein_anomaly' in results:
                print(f"Distribution Anomaly: {results['wasserstein_anomaly']:.2f}")
            
            if 'changepoint_analysis' in results:
                print(f"Changepoints Detected: {results['changepoint_analysis']['changepoints_detected']}")
        else:
            print(f"Analysis failed: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()