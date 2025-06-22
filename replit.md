# Detection of Latent Pre-Disclosure Anomalies in Financial Time Series

## Overview

This is a professional academic research system focused on "Detection of Latent Pre-Disclosure Anomalies in Financial Time Series via Supervised Learning and Signal Compression Techniques". The system applies sophisticated anomaly detection algorithms to identify pre-disclosure events in financial markets, featuring both command-line and web-based interfaces with professional visualizations for academic research purposes.

## System Architecture

### Core Components
- **Main Monitor**: Simple end-to-end system (`insider_monitor.py`) with automatic alerts
- **Advanced Analyzer**: Enhanced analysis engine with machine learning features
- **Interactive Interface**: Command-line menu system for easy operation
- **Real-time Data**: Yahoo Finance API integration for authentic market data
- **Email Alerts**: Automatic notifications for high-risk anomalies

### Application Structure
```
├── insider_monitor.py (Main end-to-end system with alerts)
├── simple_analyzer.py (Advanced analysis engine)
├── run_analysis.py (Interactive interface)
├── alert_system.py (Email/SMS notification system)
├── real_time_monitor.py (Continuous monitoring)
├── insider_trading_detector.py (Advanced ML components - optional)
├── utils.py (Utility functions)
├── README.md (Complete documentation)
└── pyproject.toml (Python dependencies)
```

## Key Components

### 1. Simple Analyzer (simple_analyzer.py)
- **Purpose**: Streamlined anomaly detection using statistical methods
- **Features**: Volume spike detection, price movement analysis, anomaly scoring
- **Data Sources**: Yahoo Finance API for authentic market data
- **Algorithms**: Z-score analysis, statistical thresholds, volume ratio analysis

### 2. Interactive Interface (run_analysis.py)
- **Purpose**: User-friendly command-line interface for analysis
- **Features**: Menu-driven operation, custom stock selection, automated monitoring
- **Reports**: Clean text-based summaries with key insights

### 3. Analysis Methods
- **Volume Anomalies**: Detects unusual trading volume (3+ standard deviations)
- **Price Anomalies**: Identifies significant price movements (5%+ daily changes)
- **Pattern Recognition**: Combines volume and price signals for comprehensive scoring
- **Recent Activity**: Monitors latest trading patterns for immediate alerts

### 4. Alert System (alert_system.py)
- **Purpose**: Real-time email and SMS notifications for high-risk anomalies
- **Features**: SendGrid email integration, Twilio SMS support, configurable thresholds
- **Triggers**: Automatically sends alerts when anomaly scores exceed 5.0
- **Free Tiers**: SendGrid (100 emails/day), Twilio (free trial credits)

### 5. Real-time Monitor (real_time_monitor.py)
- **Purpose**: Continuous monitoring system for live anomaly detection
- **Features**: Scheduled checks, customizable watchlists, session summaries
- **Monitoring**: Configurable intervals (default 30 minutes), persistent logging

## Data Flow

1. **Data Collection**: Stock data fetched via Yahoo Finance API for selected symbols
2. **Statistical Analysis**: Volume and price patterns analyzed using Z-scores and thresholds
3. **Anomaly Detection**: Unusual trading volume (3+ std dev) and significant price moves (5%+) flagged
4. **Scoring System**: Combined anomaly score calculated from volume and price signals
5. **Pattern Recognition**: Recent activity monitored for immediate alerts
6. **Text Reports**: Clean, readable summaries with key insights and top anomalies

## Usage

### Quick Start
```bash
python simple_analyzer.py
```

### Interactive Menu
```bash
python run_analysis.py
```

### Available Analysis Types
- Quick Analysis: 5 major stocks over 90 days
- Custom Analysis: User-selected stocks and timeframes
- Recent Activity: Last 7 days monitoring
- Full Report: Comprehensive 10-stock analysis over 1 year

### Real-time Monitoring
```bash
python real_time_monitor.py
```

### Alert Configuration
Set environment variables for notifications:
```bash
# Email alerts (SendGrid - 100 free emails/day)
export SENDGRID_API_KEY="your_sendgrid_api_key"
export ALERT_EMAIL="your@email.com"

# SMS alerts (Twilio - free trial credits)
export TWILIO_ACCOUNT_SID="your_account_sid"
export TWILIO_AUTH_TOKEN="your_auth_token"
export TWILIO_PHONE_NUMBER="+1234567890"
export ALERT_SMS="+1987654321"
```

## External Dependencies

### Core Libraries
- **Flask**: Web framework for API and UI
- **yfinance**: Yahoo Finance API for stock data
- **scikit-learn**: Machine learning algorithms
- **pandas/numpy**: Data processing and analysis
- **plotly**: Interactive data visualizations

### Data Sources
- **Yahoo Finance**: Primary source for historical stock data
- **API Keys**: Optional integration with Alpha Vantage, Finnhub, Polygon APIs
- **No Database**: Currently uses in-memory data storage with caching

### System Dependencies
- **Python 3.11**: Runtime environment
- **Nix Packages**: Cairo, FFmpeg, GTK3 for plotting and visualization support

## Deployment Strategy

### Current Setup
- **Platform**: Replit with Nix package management
- **Runtime**: Python 3.11 with Flask development server
- **Port**: 5000 (Flask default)
- **Auto-install**: Dependencies installed on startup via pip

### Production Considerations
- **WSGI Server**: Should use Gunicorn or uWSGI for production
- **Database**: Consider adding PostgreSQL for persistent data storage
- **Caching**: Redis integration for improved performance
- **Load Balancing**: Multiple worker processes for scalability
- **Monitoring**: Application performance monitoring and logging

### Environment Variables
- `ALPHA_VANTAGE_API_KEY`: Optional API key for additional data sources
- `FINNHUB_API_KEY`: Optional Finnhub integration
- `POLYGON_API_KEY`: Optional Polygon.io integration
- `SECRET_KEY`: Flask application secret key
- `DEBUG`: Debug mode toggle

## Changelog
- June 22, 2025. Initial setup
- June 22, 2025. Added comprehensive alert system with email notifications
- June 22, 2025. Created simple end-to-end monitor (insider_monitor.py) with automatic alerts
- June 22, 2025. Implemented real-time monitoring and continuous surveillance capabilities
- June 22, 2025. Transformed to professional academic research system "Detection of Latent Pre-Disclosure Anomalies in Financial Time Series via Supervised Learning and Signal Compression Techniques"
- June 22, 2025. Added Chart.js visualizations with anomaly score distribution and signal compression timeline charts
- June 22, 2025. Updated all terminology from insider trading to academic research language (pre-disclosure events, signal compression, latent anomalies)
- June 22, 2025. Implemented novel algorithmic methods directly in web_app.py: entropy-based complexity analysis, Wasserstein-inspired distribution distance, information-theoretic coupling, and adaptive KDE anomaly detection
- June 22, 2025. Updated README.md to showcase advanced capabilities and superior methodologies
- June 22, 2025. Enhanced web dashboard text readability with darker colors for novel research methodology descriptions

## User Preferences

Preferred communication style: Professional quantitative finance terminology.
Focus: Professional academic research interface with sophisticated algorithmic explanations, trading signal intelligence, and quantitative metrics. Avoid any references to insider trading or illegal activities - focus on statistical anomaly detection and market microstructure analysis.