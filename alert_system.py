#!/usr/bin/env python3
"""
Real-time Alert System for Insider Trading Detection
Supports email (SendGrid) and SMS (Twilio) notifications
"""

import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Email support
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Email, To, Content
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

# SMS support  
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

class AlertSystem:
    """Handles email and SMS alerts for trading anomalies"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Email configuration
        self.sendgrid_key = os.environ.get('SENDGRID_API_KEY')
        self.from_email = os.environ.get('ALERT_FROM_EMAIL', 'alerts@insider-trading-detector.com')
        
        # SMS configuration  
        self.twilio_sid = os.environ.get('TWILIO_ACCOUNT_SID')
        self.twilio_token = os.environ.get('TWILIO_AUTH_TOKEN')
        self.twilio_phone = os.environ.get('TWILIO_PHONE_NUMBER')
        
        # User preferences (can be configured)
        self.alert_recipients = {
            'email': os.environ.get('ALERT_EMAIL', '').split(',') if os.environ.get('ALERT_EMAIL') else [],
            'sms': os.environ.get('ALERT_SMS', '').split(',') if os.environ.get('ALERT_SMS') else []
        }
        
        # Alert thresholds
        self.high_risk_threshold = 5.0  # Anomaly score threshold for alerts
        self.alert_cooldown = {}  # Prevent spam alerts for same symbol
        
        self.validate_services()
    
    def setup_logging(self):
        """Setup logging for alert system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('alerts.log'),
                logging.StreamHandler()
            ]
        )
    
    def validate_services(self):
        """Check which notification services are available"""
        self.email_enabled = bool(SENDGRID_AVAILABLE and self.sendgrid_key)
        self.sms_enabled = bool(TWILIO_AVAILABLE and self.twilio_sid and self.twilio_token and self.twilio_phone)
        
        status = []
        if self.email_enabled:
            status.append("✅ Email alerts enabled")
        else:
            status.append("❌ Email alerts disabled (missing SendGrid API key)")
            
        if self.sms_enabled:
            status.append("✅ SMS alerts enabled")
        else:
            status.append("❌ SMS alerts disabled (missing Twilio credentials)")
        
        self.logger.info("Alert system status:")
        for s in status:
            self.logger.info(s)
    
    def configure_recipients(self, email_list: List[str] = None, sms_list: List[str] = None):
        """Configure alert recipients"""
        if email_list:
            self.alert_recipients['email'] = email_list
            self.logger.info(f"Email recipients configured: {len(email_list)} addresses")
        
        if sms_list:
            self.alert_recipients['sms'] = sms_list
            self.logger.info(f"SMS recipients configured: {len(sms_list)} numbers")
    
    def should_send_alert(self, symbol: str, anomaly_score: float) -> bool:
        """Check if alert should be sent based on cooldown and threshold"""
        if anomaly_score < self.high_risk_threshold:
            return False
        
        # Check cooldown (don't spam same symbol within 1 hour)
        now = datetime.now()
        cooldown_key = f"{symbol}_{now.strftime('%Y%m%d_%H')}"
        
        if cooldown_key in self.alert_cooldown:
            return False
            
        self.alert_cooldown[cooldown_key] = now
        return True
    
    def send_email_alert(self, subject: str, content: str) -> bool:
        """Send email alert using SendGrid"""
        if not self.email_enabled or not self.alert_recipients['email']:
            return False
        
        try:
            sg = SendGridAPIClient(self.sendgrid_key)
            
            for recipient in self.alert_recipients['email']:
                message = Mail(
                    from_email=Email(self.from_email),
                    to_emails=To(recipient),
                    subject=subject,
                    html_content=Content("text/html", content)
                )
                
                response = sg.send(message)
                self.logger.info(f"Email sent to {recipient}: {response.status_code}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Email sending failed: {e}")
            return False
    
    def send_sms_alert(self, message: str) -> bool:
        """Send SMS alert using Twilio"""
        if not self.sms_enabled or not self.alert_recipients['sms']:
            return False
        
        try:
            client = TwilioClient(self.twilio_sid, self.twilio_token)
            
            for recipient in self.alert_recipients['sms']:
                sms = client.messages.create(
                    body=message,
                    from_=self.twilio_phone,
                    to=recipient
                )
                self.logger.info(f"SMS sent to {recipient}: {sms.sid}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"SMS sending failed: {e}")
            return False
    
    def format_alert_content(self, anomaly_data: Dict) -> tuple:
        """Format alert content for email and SMS"""
        symbol = anomaly_data.get('symbol', 'UNKNOWN')
        score = anomaly_data.get('anomaly_score', 0)
        price_change = anomaly_data.get('price_change', 0)
        volume_ratio = anomaly_data.get('volume_ratio', 1)
        date = anomaly_data.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        # Email content (HTML)
        email_subject = f"🚨 HIGH RISK ALERT: {symbol} - Anomaly Score {score:.2f}"
        
        email_content = f"""
        <html>
        <body>
            <h2 style="color: #d32f2f;">🚨 Insider Trading Alert</h2>
            <div style="background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0;">
                <h3>High-Risk Anomaly Detected</h3>
                <p><strong>Symbol:</strong> {symbol}</p>
                <p><strong>Date:</strong> {date}</p>
                <p><strong>Anomaly Score:</strong> {score:.2f} (Threshold: {self.high_risk_threshold})</p>
                <p><strong>Price Change:</strong> {price_change:+.2f}%</p>
                <p><strong>Volume Ratio:</strong> {volume_ratio:.1f}x normal</p>
            </div>
            <p style="color: #666; font-size: 12px;">
                This alert was generated by the Insider Trading Detection System.
                Please review the trading activity and consider further investigation.
            </p>
        </body>
        </html>
        """
        
        # SMS content (plain text, short)
        sms_content = f"🚨 ALERT: {symbol} anomaly score {score:.1f}. Price {price_change:+.1f}%, Volume {volume_ratio:.1f}x. {date}"
        
        return email_subject, email_content, sms_content
    
    def send_anomaly_alert(self, anomaly_data: Dict) -> Dict[str, bool]:
        """Send alert for detected anomaly"""
        symbol = anomaly_data.get('symbol', 'UNKNOWN')
        score = anomaly_data.get('anomaly_score', 0)
        
        if not self.should_send_alert(symbol, score):
            return {'email': False, 'sms': False, 'reason': 'threshold_not_met_or_cooldown'}
        
        # Format alert content
        email_subject, email_content, sms_content = self.format_alert_content(anomaly_data)
        
        # Send alerts
        results = {
            'email': self.send_email_alert(email_subject, email_content),
            'sms': self.send_sms_alert(sms_content),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log alert
        self.logger.warning(f"HIGH RISK ALERT SENT: {symbol} (Score: {score:.2f})")
        
        return results
    
    def test_alerts(self) -> Dict[str, bool]:
        """Test alert system with sample data"""
        test_data = {
            'symbol': 'TEST',
            'anomaly_score': 8.5,
            'price_change': 12.3,
            'volume_ratio': 3.2,
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        
        self.logger.info("Testing alert system...")
        results = self.send_anomaly_alert(test_data)
        
        return results

def main():
    """Test the alert system"""
    print("🚨 Testing Alert System")
    print("=" * 50)
    
    # Initialize alert system
    alert_system = AlertSystem()
    
    # Configure test recipients if not set via environment
    if not alert_system.alert_recipients['email']:
        email = input("Enter test email address (or press Enter to skip): ").strip()
        if email:
            alert_system.configure_recipients(email_list=[email])
    
    if not alert_system.alert_recipients['sms']:
        sms = input("Enter test SMS number with country code (e.g., +1234567890, or press Enter to skip): ").strip()
        if sms:
            alert_system.configure_recipients(sms_list=[sms])
    
    # Test alerts
    if alert_system.alert_recipients['email'] or alert_system.alert_recipients['sms']:
        results = alert_system.test_alerts()
        print("\nTest Results:")
        print(f"Email: {'✅ Sent' if results['email'] else '❌ Failed'}")
        print(f"SMS: {'✅ Sent' if results['sms'] else '❌ Failed'}")
    else:
        print("No recipients configured. Set environment variables:")
        print("ALERT_EMAIL=your@email.com")
        print("ALERT_SMS=+1234567890")

if __name__ == "__main__":
    main()