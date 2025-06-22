#!/usr/bin/env python3
"""
Simple Insider Trading Monitor
Runs analysis and sends alerts - nothing fancy
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('insider_monitor.log'),
        logging.StreamHandler()
    ]
)

class SimpleInsiderMonitor:
    def __init__(self):
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        self.alert_threshold = 5.0
        self.email_alerts = os.getenv('ALERT_EMAIL')
        self.smtp_settings = {
            'server': 'smtp.gmail.com',
            'port': 587,
            'username': os.getenv('EMAIL_USERNAME'),
            'password': os.getenv('EMAIL_PASSWORD')
        }
    
    def fetch_data(self, symbol, days=90):
        """Fetch stock data"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                logging.warning(f"No data for {symbol}")
                return None
                
            return data
        except Exception as e:
            logging.error(f"Error fetching {symbol}: {e}")
            return None
    
    def analyze_anomalies(self, data, symbol):
        """Simple anomaly detection"""
        if data is None or len(data) < 10:
            return []
        
        # Calculate daily returns
        data['Returns'] = data['Close'].pct_change()
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
        
        # Z-scores for volume
        volume_mean = data['Volume'].mean()
        volume_std = data['Volume'].std()
        data['Volume_ZScore'] = (data['Volume'] - volume_mean) / volume_std
        
        anomalies = []
        
        for i, row in data.iterrows():
            if pd.isna(row['Returns']) or pd.isna(row['Volume_Ratio']):
                continue
                
            # Anomaly scoring
            score = 0
            
            # Large price movements (>3%)
            if abs(row['Returns']) > 0.03:
                score += abs(row['Returns']) * 100
            
            # High volume (>2 std devs)
            if row['Volume_ZScore'] > 2:
                score += row['Volume_ZScore']
            
            # Volume spike with price movement
            if row['Volume_Ratio'] > 2 and abs(row['Returns']) > 0.02:
                score += row['Volume_Ratio'] * 2
            
            if score > 2.0:  # Threshold for recording anomaly
                anomalies.append({
                    'date': i.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'score': round(score, 2),
                    'return': round(row['Returns'] * 100, 2),
                    'volume_ratio': round(row['Volume_Ratio'], 1),
                    'volume_zscore': round(row['Volume_ZScore'], 2)
                })
        
        return anomalies
    
    def send_email_alert(self, subject, body):
        """Send email alert"""
        if not self.email_alerts or not self.smtp_settings['username']:
            logging.info("Email alerts not configured")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_settings['username']
            msg['To'] = self.email_alerts
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_settings['server'], self.smtp_settings['port'])
            server.starttls()
            server.login(self.smtp_settings['username'], self.smtp_settings['password'])
            server.send_message(msg)
            server.quit()
            
            logging.info(f"Email alert sent: {subject}")
            return True
            
        except Exception as e:
            logging.error(f"Email failed: {e}")
            return False
    
    def run_analysis(self):
        """Run complete analysis"""
        logging.info("Starting insider trading analysis...")
        
        all_anomalies = []
        high_risk_alerts = []
        
        for symbol in self.symbols:
            logging.info(f"Analyzing {symbol}...")
            
            data = self.fetch_data(symbol)
            anomalies = self.analyze_anomalies(data, symbol)
            
            if anomalies:
                all_anomalies.extend(anomalies)
                
                # Check for high-risk alerts
                for anomaly in anomalies:
                    if anomaly['score'] >= self.alert_threshold:
                        high_risk_alerts.append(anomaly)
                        logging.warning(f"HIGH RISK: {symbol} score {anomaly['score']}")
        
        # Generate report
        self.generate_report(all_anomalies)
        
        # Send alerts for high-risk anomalies
        if high_risk_alerts:
            self.send_high_risk_alerts(high_risk_alerts)
        
        logging.info(f"Analysis complete. Found {len(all_anomalies)} anomalies, {len(high_risk_alerts)} high-risk")
        
        return all_anomalies, high_risk_alerts
    
    def generate_report(self, anomalies):
        """Generate text report"""
        print("\n" + "="*60)
        print("INSIDER TRADING DETECTION REPORT")
        print("="*60)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Symbols Analyzed: {', '.join(self.symbols)}")
        print(f"Total Anomalies: {len(anomalies)}")
        
        if anomalies:
            # Sort by score
            sorted_anomalies = sorted(anomalies, key=lambda x: x['score'], reverse=True)
            
            print(f"\nTop 10 Anomalies:")
            print("-" * 60)
            for i, anomaly in enumerate(sorted_anomalies[:10], 1):
                print(f"{i:2d}. {anomaly['symbol']} on {anomaly['date']}")
                print(f"    Score: {anomaly['score']} | Return: {anomaly['return']:+.1f}%")
                print(f"    Volume: {anomaly['volume_ratio']}x normal (Z-score: {anomaly['volume_zscore']})")
            
            # Recent activity (last 7 days)
            recent_date = datetime.now() - timedelta(days=7)
            recent_anomalies = [a for a in anomalies if datetime.strptime(a['date'], '%Y-%m-%d') > recent_date]
            
            if recent_anomalies:
                print(f"\nRecent Activity (Last 7 days): {len(recent_anomalies)} anomalies")
                print("-" * 60)
                for anomaly in recent_anomalies:
                    print(f"{anomaly['symbol']} on {anomaly['date']}: Score {anomaly['score']}")
        
        print("="*60)
    
    def send_high_risk_alerts(self, alerts):
        """Send alerts for high-risk anomalies"""
        if not alerts:
            return
        
        subject = f"HIGH RISK: Insider Trading Alert - {len(alerts)} anomalies detected"
        
        body = f"""INSIDER TRADING ALERT
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

High-risk anomalies detected (score >= {self.alert_threshold}):

"""
        
        for alert in alerts:
            body += f"""
{alert['symbol']} on {alert['date']}
- Anomaly Score: {alert['score']}
- Price Change: {alert['return']:+.1f}%
- Volume: {alert['volume_ratio']}x normal
- Volume Z-Score: {alert['volume_zscore']}
"""
        
        body += f"""
Total high-risk alerts: {len(alerts)}

This is an automated alert from the Insider Trading Detection System.
"""
        
        # Log alert even if email fails
        logging.warning(f"HIGH RISK ALERT: {len(alerts)} anomalies detected")
        for alert in alerts:
            logging.warning(f"  {alert['symbol']}: Score {alert['score']}")
        
        # Send email
        self.send_email_alert(subject, body)

def main():
    """Main function"""
    print("Simple Insider Trading Monitor")
    print("=" * 40)
    
    monitor = SimpleInsiderMonitor()
    anomalies, high_risk = monitor.run_analysis()
    
    print(f"\nSummary:")
    print(f"- Total anomalies found: {len(anomalies)}")
    print(f"- High-risk alerts: {len(high_risk)}")
    print(f"- Log file: insider_monitor.log")
    
    if high_risk:
        print(f"- Email alerts sent: {'Yes' if monitor.email_alerts else 'No (not configured)'}")

if __name__ == "__main__":
    main()