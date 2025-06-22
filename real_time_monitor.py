#!/usr/bin/env python3
"""
Real-time Insider Trading Anomaly Monitor
Continuously monitors stocks and sends alerts for suspicious activity
"""

import time
import schedule
from datetime import datetime, timedelta
from simple_analyzer import SimpleInsiderTradingDetector
import logging

class RealTimeMonitor:
    """Real-time monitoring system for insider trading anomalies"""
    
    def __init__(self, watchlist=None, check_interval=30):
        """
        Initialize the real-time monitor
        
        Args:
            watchlist: List of stock symbols to monitor
            check_interval: Minutes between checks
        """
        self.watchlist = watchlist or ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META']
        self.check_interval = check_interval
        self.detector = SimpleInsiderTradingDetector(enable_alerts=True)
        self.last_check = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('real_time_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        print(f"Real-time monitor initialized for {len(self.watchlist)} stocks")
        print(f"Check interval: {check_interval} minutes")
        print(f"Watchlist: {', '.join(self.watchlist)}")
    
    def configure_alerts(self, email_list=None, sms_list=None):
        """Configure alert recipients"""
        self.detector.configure_alerts(email_list, sms_list)
    
    def check_recent_activity(self):
        """Check for recent trading anomalies"""
        print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Checking for anomalies...")
        
        try:
            # Analyze recent data (last 2 days to catch intraday movements)
            results = self.detector.analyze_stocks(self.watchlist, days_back=2)
            
            if results is not None:
                # Focus on today's activity
                today = datetime.now().date()
                yesterday = today - timedelta(days=1)
                
                recent_data = results[results.index.date >= yesterday]
                high_risk = recent_data[recent_data['Anomaly_Score'] > 5.0]
                
                if len(high_risk) > 0:
                    print(f"HIGH RISK ALERTS: {len(high_risk)} anomalies detected!")
                    for date, row in high_risk.iterrows():
                        print(f"  {row['Symbol']} on {date.strftime('%Y-%m-%d')}: Score {row['Anomaly_Score']:.2f}")
                else:
                    print("No high-risk anomalies detected")
                
                # Update last check time
                self.last_check[datetime.now().strftime('%Y-%m-%d %H:%M')] = len(high_risk)
                
        except Exception as e:
            self.logger.error(f"Error during monitoring check: {e}")
            print(f"Error during check: {e}")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        print(f"\nStarting real-time monitoring...")
        print(f"Press Ctrl+C to stop")
        print("=" * 60)
        
        # Schedule regular checks
        schedule.every(self.check_interval).minutes.do(self.check_recent_activity)
        
        # Run initial check
        self.check_recent_activity()
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute for scheduled tasks
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            self.show_session_summary()
    
    def show_session_summary(self):
        """Show summary of monitoring session"""
        print("\nMonitoring Session Summary:")
        print("-" * 40)
        
        if self.last_check:
            total_alerts = sum(self.last_check.values())
            print(f"Total checks: {len(self.last_check)}")
            print(f"Total high-risk alerts: {total_alerts}")
            print(f"Watchlist: {', '.join(self.watchlist)}")
            
            if self.last_check:
                print("\nRecent checks:")
                for check_time, alert_count in list(self.last_check.items())[-5:]:
                    status = f"{alert_count} alerts" if alert_count > 0 else "No alerts"
                    print(f"  {check_time}: {status}")
        else:
            print("No checks completed")

def main():
    """Main function for real-time monitoring"""
    print("Real-Time Insider Trading Monitor")
    print("=" * 50)
    
    # Configuration options
    print("\nConfiguration Options:")
    print("1. Use default watchlist (AAPL, GOOGL, MSFT, AMZN, TSLA, NVDA, META)")
    print("2. Custom watchlist")
    print("3. Quick test (single check)")
    
    try:
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == "2":
            symbols_input = input("Enter stock symbols (comma-separated): ").strip().upper()
            watchlist = [s.strip() for s in symbols_input.split(',') if s.strip()]
        elif choice == "3":
            # Quick test mode
            print("\nRunning quick test...")
            detector = SimpleInsiderTradingDetector(enable_alerts=True)
            detector.analyze_stocks(['AAPL', 'TSLA'], days_back=7)
            return
        else:
            watchlist = None  # Use default
        
        # Set check interval
        interval_input = input("Check interval in minutes (default 30): ").strip()
        interval = int(interval_input) if interval_input.isdigit() else 30
        
        # Initialize monitor
        monitor = RealTimeMonitor(watchlist, interval)
        
        # Configure alerts if desired
        email_input = input("Email for alerts (optional): ").strip()
        sms_input = input("SMS number for alerts (optional, format: +1234567890): ").strip()
        
        if email_input or sms_input:
            email_list = [email_input] if email_input else None
            sms_list = [sms_input] if sms_input else None
            monitor.configure_alerts(email_list, sms_list)
        
        # Start monitoring
        monitor.start_monitoring()
        
    except KeyboardInterrupt:
        print("\nSetup cancelled by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()