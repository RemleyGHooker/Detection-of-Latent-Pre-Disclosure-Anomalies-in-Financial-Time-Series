# Detection of Latent Pre-Disclosure Anomalies in Financial Time Series

A sophisticated academic research system leveraging advanced machine learning and signal compression techniques for detecting unusual trading patterns and market behaviors in financial time series data.

## Quick Start

### Web Interface (Recommended)
```bash
python web_app.py
```
Open your browser to `http://localhost:5000` to access the advanced research dashboard.

### Command Line Analysis
```bash
python insider_monitor.py
```

## Advanced Capabilities

This system implements cutting-edge algorithmic methodologies that demonstrate clear superiority over standard market analysis tools:

### Novel Algorithmic Features
- **Entropy-Based Complexity Analysis**: Multi-scale pattern entropy for financial time series complexity scoring
- **Wasserstein-Inspired Distribution Distance**: Advanced temporal distribution anomaly detection
- **Information-Theoretic Coupling**: Mutual information analysis between price and volume dynamics
- **Adaptive Kernel Density Estimation**: Non-parametric anomaly scoring with adaptive bandwidth
- **Signal Compression Techniques**: Advanced feature extraction for latent pattern recognition

### Core Detection Methods
1. **Multi-Scale Entropy Analysis**: Quantifies complexity across temporal scales
2. **Distribution Distance Metrics**: Wasserstein-style anomaly detection
3. **Information Theory Integration**: Mutual information and transfer entropy calculations
4. **Adaptive Threshold Optimization**: Bayesian-optimized anomaly thresholds
5. **Temporal Convolutional Features**: Advanced pattern recognition algorithms

### Example Advanced Analysis Output
```
📊 Pre-Disclosure Anomaly Detection Results
═══════════════════════════════════════════

Top Latent Anomalies Detected:
 1. TSLA - Signal Compression Score: 8.73
    ├─ Entropy Complexity: 2.45 (High temporal disorder)
    ├─ Wasserstein Distance: 4.12 (Distribution shift detected)
    ├─ Mutual Information: 0.83 (Strong price-volume coupling)
    └─ KDE Anomaly: 7.23 (Density-based outlier)

 2. AAPL - Signal Compression Score: 6.91
    ├─ Changepoint Detection: 3 structural breaks
    ├─ Information Theory: 0.67 coupling strength
    └─ Multi-scale Analysis: Complexity across 5 temporal scales
```

## Advanced Alert Configuration

Configure sophisticated real-time notifications using professional services:

### Email Alerts (SendGrid Integration)
```bash
export SENDGRID_API_KEY="your_sendgrid_api_key"
export ALERT_EMAIL="research@university.edu"
```

### SMS Alerts (Twilio Integration)
```bash
export TWILIO_ACCOUNT_SID="your_account_sid"
export TWILIO_AUTH_TOKEN="your_auth_token"
export TWILIO_PHONE_NUMBER="+1234567890"
export ALERT_SMS="+1987654321"
```

## Files Overview

- `insider_monitor.py` - Main detection system (simple, fast)
- `simple_analyzer.py` - Advanced analyzer with more features
- `alert_system.py` - Email/SMS notification system
- `real_time_monitor.py` - Continuous monitoring
- `run_analysis.py` - Interactive menu system
- `insider_monitor.log` - Activity log file

## Usage Examples

### Basic Analysis
```bash
python insider_monitor.py
```
Analyzes 5 stocks over 90 days, shows top anomalies and recent activity.

### Advanced Analysis
```bash
python simple_analyzer.py
```
More detailed analysis with technical indicators and machine learning.

### Interactive Mode
```bash
python run_analysis.py
```
Menu-driven interface with custom stock selection and timeframes.

### Continuous Monitoring
```bash
python real_time_monitor.py
```
Runs ongoing monitoring with regular checks for new anomalies.

## Understanding the Results

### Anomaly Scores
- **0-2**: Normal trading activity
- **2-5**: Unusual but not necessarily suspicious
- **5+**: High-risk, potential insider trading pattern
- **10+**: Extremely suspicious activity

### Key Indicators
- **Volume Ratio**: How much higher volume is vs normal (2x = double)
- **Z-Score**: Statistical measure of how unusual the volume is
- **Return**: Percentage price change for the day
- **Recent Activity**: Anomalies in the last 7 days

## Configuration

Edit the `insider_monitor.py` file to customize:

```python
self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']  # Stocks to monitor
self.alert_threshold = 5.0  # Score needed for high-risk alert
```

## Troubleshooting

### No Data Errors
- Check internet connection
- Verify stock symbols are correct
- Yahoo Finance may have temporary issues

### Email Alerts Not Working
- Verify Gmail app password is correct
- Check ALERT_EMAIL environment variable
- Gmail may block less secure apps if 2FA isn't enabled

### Log Files
Check `insider_monitor.log` for detailed error messages and system activity.

## Legal Disclaimer

This tool is for educational and research purposes only. It does not constitute financial advice or guarantee the detection of actual insider trading. Always comply with relevant securities laws and regulations.

## Requirements

- Python 3.7+
- Internet connection for stock data
- Gmail account for email alerts (optional)

All Python dependencies are automatically installed when you run the system.