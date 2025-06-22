#!/usr/bin/env python3
"""
Simple Insider Trading Detection Analyzer
A streamlined version focused on core functionality
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from alert_system import AlertSystem
warnings.filterwarnings('ignore')

class SimpleInsiderTradingDetector:
    def __init__(self, enable_alerts=True):
        self.results = None
        self.alert_system = AlertSystem() if enable_alerts else None
        if self.alert_system:
            print("🚨 Alert system initialized")
        
    def analyze_stocks(self, symbols=['AAPL', 'GOOGL', 'MSFT'], days_back=365):
        """
        Analyze stocks for potential insider trading patterns
        """
        print(f"🔍 Analyzing {len(symbols)} stocks for insider trading patterns...")
        print(f"📅 Looking back {days_back} days")
        print("-" * 60)
        
        # Get data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        all_data = []
        
        for symbol in symbols:
            print(f"📊 Fetching data for {symbol}...")
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if len(data) < 50:
                    print(f"⚠️  Insufficient data for {symbol}")
                    continue
                
                # Calculate features
                data['Symbol'] = symbol
                data['Returns'] = data['Close'].pct_change()
                data['Volume_MA'] = data['Volume'].rolling(20).mean()
                data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
                
                # Calculate volume z-score
                data['Volume_Std'] = data['Volume'].rolling(20).std()
                data['Volume_Z'] = (data['Volume'] - data['Volume_MA']) / data['Volume_Std']
                
                # Calculate price volatility
                data['Volatility'] = data['Returns'].rolling(20).std()
                
                # Detect anomalies (simple threshold-based)
                volume_threshold = 3.0  # 3 standard deviations
                return_threshold = 0.05  # 5% daily return
                
                data['Volume_Anomaly'] = (abs(data['Volume_Z']) > volume_threshold).astype(int)
                data['Price_Anomaly'] = (abs(data['Returns']) > return_threshold).astype(int)
                data['Combined_Anomaly'] = (data['Volume_Anomaly'] | data['Price_Anomaly']).astype(int)
                
                # Calculate anomaly score
                data['Anomaly_Score'] = (
                    abs(data['Volume_Z'].fillna(0)) * 0.6 + 
                    abs(data['Returns'].fillna(0)) * 100 * 0.4
                )
                
                all_data.append(data)
                print(f"✅ {symbol}: {len(data)} days, {data['Combined_Anomaly'].sum()} anomalies detected")
                
                # Check for high-risk anomalies and send alerts
                if self.alert_system:
                    self._check_and_send_alerts(data, symbol)
                
            except Exception as e:
                print(f"❌ Error with {symbol}: {e}")
        
        if not all_data:
            print("❌ No data collected!")
            return None
        
        # Combine all data
        self.results = pd.concat(all_data)
        print(f"\n📈 Total observations: {len(self.results):,}")
        print(f"🚨 Total anomalies: {self.results['Combined_Anomaly'].sum():,}")
        print(f"📊 Anomaly rate: {self.results['Combined_Anomaly'].mean():.2%}")
        
        return self.results
    
    def show_top_anomalies(self, n=10):
        """Show the top anomalies detected"""
        if self.results is None:
            print("❌ No analysis results available. Run analyze_stocks() first.")
            return
        
        print(f"\n🔥 Top {n} Anomalies by Score:")
        print("-" * 80)
        
        top_anomalies = self.results.nlargest(n, 'Anomaly_Score')
        
        for i, (date, row) in enumerate(top_anomalies.iterrows(), 1):
            print(f"{i:2d}. {row['Symbol']} on {date.strftime('%Y-%m-%d')}")
            print(f"    📊 Anomaly Score: {row['Anomaly_Score']:.2f}")
            print(f"    📈 Price Change: {row['Returns']*100:+.2f}%")
            print(f"    📦 Volume Ratio: {row['Volume_Ratio']:.1f}x normal")
            print(f"    🎯 Volume Z-Score: {row['Volume_Z']:+.2f}")
            print()
    
    def show_symbol_summary(self):
        """Show summary by symbol"""
        if self.results is None:
            print("❌ No analysis results available. Run analyze_stocks() first.")
            return
        
        print("\n📋 Symbol Summary:")
        print("-" * 60)
        
        summary = self.results.groupby('Symbol').agg({
            'Combined_Anomaly': ['count', 'sum'],
            'Anomaly_Score': ['mean', 'max'],
            'Volume_Ratio': 'mean',
            'Returns': lambda x: x.abs().mean()
        }).round(3)
        
        summary.columns = ['Days', 'Anomalies', 'Avg Score', 'Max Score', 'Avg Vol Ratio', 'Avg |Return|']
        summary['Anomaly Rate'] = (summary['Anomalies'] / summary['Days']).round(3)
        
        print(summary.to_string())
    
    def show_recent_activity(self, days=7):
        """Show recent suspicious activity"""
        if self.results is None:
            print("❌ No analysis results available. Run analyze_stocks() first.")
            return
        
        cutoff_date = self.results.index.max() - timedelta(days=days)
        recent = self.results[self.results.index >= cutoff_date]
        recent_anomalies = recent[recent['Combined_Anomaly'] == 1]
        
        print(f"\n⏰ Recent Activity (Last {days} days):")
        print("-" * 60)
        
        if len(recent_anomalies) == 0:
            print("✅ No anomalies detected in recent activity")
            return
        
        for date, row in recent_anomalies.iterrows():
            print(f"🚨 {row['Symbol']} on {date.strftime('%Y-%m-%d')}")
            print(f"    Score: {row['Anomaly_Score']:.2f} | Return: {row['Returns']*100:+.2f}% | Volume: {row['Volume_Ratio']:.1f}x")
    
    def simple_report(self):
        """Generate a simple text report"""
        if self.results is None:
            print("❌ No analysis results available. Run analyze_stocks() first.")
            return
        
        print("\n" + "="*80)
        print("📊 INSIDER TRADING DETECTION REPORT")
        print("="*80)
        
        # Overall stats
        total_obs = len(self.results)
        total_anomalies = self.results['Combined_Anomaly'].sum()
        anomaly_rate = self.results['Combined_Anomaly'].mean()
        avg_score = self.results['Anomaly_Score'].mean()
        
        print(f"📈 Total Observations: {total_obs:,}")
        print(f"🚨 Total Anomalies: {total_anomalies:,}")
        print(f"📊 Anomaly Rate: {anomaly_rate:.2%}")
        print(f"⭐ Average Anomaly Score: {avg_score:.2f}")
        
        # Symbol breakdown
        self.show_symbol_summary()
        
        # Top anomalies
        self.show_top_anomalies(5)
        
        # Recent activity
        self.show_recent_activity()
        
        print("\n" + "="*80)
    
    def _check_and_send_alerts(self, data, symbol):
        """Check for high-risk anomalies and send alerts"""
        # Find high-risk anomalies (score > 5.0)
        high_risk = data[data['Anomaly_Score'] > 5.0]
        
        for date, row in high_risk.iterrows():
            anomaly_data = {
                'symbol': symbol,
                'date': date.strftime('%Y-%m-%d'),
                'anomaly_score': row['Anomaly_Score'],
                'price_change': row['Returns'] * 100,
                'volume_ratio': row['Volume_Ratio'],
                'volume_z_score': row['Volume_Z']
            }
            
            # Send alert
            result = self.alert_system.send_anomaly_alert(anomaly_data)
            if result.get('email') or result.get('sms'):
                print(f"🚨 HIGH RISK ALERT SENT: {symbol} on {date.strftime('%Y-%m-%d')} (Score: {row['Anomaly_Score']:.2f})")
    
    def configure_alerts(self, email_list=None, sms_list=None):
        """Configure alert recipients"""
        if self.alert_system:
            self.alert_system.configure_recipients(email_list, sms_list)
            print("Alert recipients configured")
        else:
            print("Alert system not enabled")
    
    def test_alert_system(self):
        """Test the alert system"""
        if self.alert_system:
            return self.alert_system.test_alerts()
        else:
            print("Alert system not enabled")
            return None


def main():
    """Main execution function"""
    print("🚨 Simple Insider Trading Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = SimpleInsiderTradingDetector()
    
    # Default symbols to analyze
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Run analysis
    results = detector.analyze_stocks(symbols, days_back=180)
    
    if results is not None:
        # Show comprehensive report
        detector.simple_report()
        
        # Interactive options
        print("\n" + "="*50)
        print("📋 Available Commands:")
        print("detector.show_top_anomalies(10)    # Show top 10 anomalies")
        print("detector.show_symbol_summary()     # Show symbol breakdown") 
        print("detector.show_recent_activity(14)  # Show last 14 days")
        print("detector.analyze_stocks(['NVDA', 'META'])  # Analyze different stocks")
        
        return detector
    else:
        print("❌ Analysis failed!")
        return None


if __name__ == "__main__":
    detector = main()