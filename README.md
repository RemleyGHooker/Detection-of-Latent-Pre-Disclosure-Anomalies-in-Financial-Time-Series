# Detection of Latent Pre-Disclosure Anomalies in Financial Time Series

This project presents a research-grade anomaly detection framework designed to identify unusual trading behavior in equity markets preceding major corporate events or regulatory disclosures. The system leverages advanced time series analysis, signal compression, and information-theoretic methods to flag latent patterns that may indicate information leakage, algorithmic exploitation, or significant structural changes in market behavior.

This tool is intended for educational and academic research use only.

## Overview

Modern markets generate high-volume time series data across price, volume, and order flow. This project addresses the challenge of identifying latent or early-stage signals that precede observable market events. It uses a modular machine learning and statistical analysis pipeline to detect potential pre-disclosure anomalies using public data.

Key capabilities include:

- Multi-scale entropy scoring of financial time series
- Mutual information analysis between volume and price
- Wasserstein-style distance metrics to detect distribution shifts
- Adaptive kernel density estimation (KDE) for probabilistic outlier detection
- Signal compression techniques for latent feature extraction

## Quick Start

### Web Interface
Run the dashboard locally using:
```bash
python web_app.py
```
Then open your browser at `http://localhost:5000` to access the interactive anomaly research interface.

### Command Line Execution
For direct CLI-based anomaly scoring:
```bash
python insider_monitor.py
```

## Core Features

### Algorithmic Techniques

- **Multi-Scale Entropy Analysis:** Captures complexity and temporal disorder in return and volume sequences.
- **Wasserstein-Inspired Metrics:** Measures temporal distribution drift between pre-event and normal trading conditions.
- **Mutual Information and Transfer Entropy:** Detects nonlinear dependencies between price and volume activity.
- **Signal Compression Scoring:** Reduces time series into compressed representations for anomaly detection.
- **Adaptive KDE:** Uses variable bandwidth to flag statistical outliers in dynamic markets.

## System Components

| File | Purpose |
|------|---------|
| `insider_monitor.py` | Core anomaly detection engine |
| `simple_analyzer.py` | Extended analysis with technical indicators |
| `run_analysis.py` | Interactive, menu-driven interface |
| `real_time_monitor.py` | Continuous data streaming and monitoring |
| `alert_system.py` | Email and SMS notification configuration |
| `insider_monitor.log` | Time-stamped logs for result tracking and debugging |

## Example Output

```
Anomaly Detection Summary

1. TSLA - Signal Compression Score: 8.73
   - Entropy Complexity: 2.45 (Elevated disorder)
   - Wasserstein Distance: 4.12 (High distribution shift)
   - Mutual Information: 0.83 (Strong volume-price coupling)
   - KDE Anomaly Score: 7.23 (Outlier behavior)

2. AAPL - Signal Compression Score: 6.91
   - Structural Breaks: 3 changepoints detected
   - Multi-Scale Complexity: Elevated on 5 temporal levels
   - Coupling Strength: 0.67 (Moderate nonlinear dependency)
```

## Real-Time Alert Configuration

### Email Alerts (SendGrid or Gmail)
```bash
export SENDGRID_API_KEY="your_key"
export ALERT_EMAIL="your_email@example.com"
```

### SMS Alerts (Twilio)
```bash
export TWILIO_ACCOUNT_SID="your_sid"
export TWILIO_AUTH_TOKEN="your_token"
export TWILIO_PHONE_NUMBER="+1234567890"
export ALERT_SMS="+1987654321"
```

Alerts are triggered when anomaly scores exceed the defined threshold.

## Interpreting Results

| Score Range | Interpretation |
|-------------|----------------|
| 0–2         | Normal behavior |
| 2–5         | Potentially irregular |
| 5–10        | High-risk anomaly |
| 10+         | Critical outlier; further analysis recommended |

Key indicators:

- **Volume Ratio:** Compares current volume to recent average
- **Z-Score:** Standard deviation-based anomaly index
- **Return:** Daily percentage change
- **Entropy Complexity:** Degree of structural randomness

## Configuration

Modify monitored tickers and thresholds in `insider_monitor.py`:
```python
self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
self.alert_threshold = 5.0
```

## Troubleshooting

### No Data Error
- Verify internet connection
- Confirm stock ticker symbols
- Note: Yahoo Finance data may occasionally time out

### Email/SMS Alerts Not Sending
- Double check your credentials and environment variables
- Ensure third-party services (e.g., Gmail, Twilio) are properly configured and enabled

### Logs
Check `insider_monitor.log` for error diagnostics and result history.

## Legal Disclaimer

This project is provided for educational and academic research purposes only. It does not constitute financial advice, trading signals, or a guarantee of actual insider trading activity. Use of this tool should comply with all applicable securities laws and institutional data policies.

## System Requirements

- Python 3.7 or higher
- Internet access for live market data
- SendGrid or Twilio credentials (optional, for alerts)
- Compatible with Windows, macOS, and Linux

## Author

Created and maintained by Remley Hooker  
Microsoft Software Engineering Intern | Quantitative Systems Researcher | Natives Rising Fellow

For collaboration inquiries or research discussion, contact via LinkedIn or GitHub.
