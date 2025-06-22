#!/usr/bin/env python3
"""
Flask web application for Insider Trading Detection System
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import logging
from datetime import datetime
import traceback
import os

from config import CONFIG
from insider_trading_detector import InsiderTradingDetector
from utils import setup_logging

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = CONFIG['app']['secret_key']

# Setup logging
logger = setup_logging(CONFIG)

# Global detector instance
detector = None

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html', config=CONFIG)

@app.route('/dashboard')
def dashboard():
    """Dashboard page showing analysis results."""
    return render_template('dashboard.html', config=CONFIG)

@app.route('/reports')
def reports():
    """Reports page showing detailed analysis."""
    return render_template('reports.html', config=CONFIG)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """API endpoint to run insider trading analysis."""
    global detector
    
    try:
        data = request.get_json() or {}
        symbols = data.get('symbols', CONFIG['data']['symbols'])
        
        # Validate symbols
        if not symbols or not isinstance(symbols, list):
            return jsonify({
                'success': False,
                'error': 'Invalid symbols provided'
            }), 400
        
        # Initialize detector
        detector = InsiderTradingDetector(CONFIG)
        
        # Run analysis
        logger.info(f"Starting analysis for symbols: {symbols}")
        results = detector.run_analysis(symbols)
        
        if not results['success']:
            return jsonify(results), 500
        
        # Convert DataFrame to serializable format
        if results['results'] is not None:
            results['results'] = results['results'].tail(100).to_dict('records')  # Limit to last 100 records
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500

@app.route('/api/visualizations/<symbol>')
def get_visualizations(symbol):
    """API endpoint to get visualizations for a symbol."""
    global detector
    
    try:
        if detector is None or detector.results is None:
            return jsonify({
                'success': False,
                'error': 'No analysis results available. Run analysis first.'
            }), 400
        
        visualizations = detector.get_visualizations(symbol if symbol != 'all' else None)
        
        return jsonify({
            'success': True,
            'visualizations': visualizations
        })
        
    except Exception as e:
        logger.error(f"Error getting visualizations: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/symbol/<symbol>')
def get_symbol_analysis(symbol):
    """API endpoint to get detailed analysis for a specific symbol."""
    global detector
    
    try:
        if detector is None or detector.results is None:
            return jsonify({
                'success': False,
                'error': 'No analysis results available. Run analysis first.'
            }), 400
        
        analysis = detector.get_symbol_analysis(symbol)
        
        if 'error' in analysis:
            return jsonify({
                'success': False,
                'error': analysis['error']
            }), 404
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        logger.error(f"Error getting symbol analysis: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/report')
def get_report():
    """API endpoint to get the full analysis report."""
    global detector
    
    try:
        if detector is None or detector.report is None:
            return jsonify({
                'success': False,
                'error': 'No analysis report available. Run analysis first.'
            }), 400
        
        return jsonify({
            'success': True,
            'report': detector.report
        })
        
    except Exception as e:
        logger.error(f"Error getting report: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status')
def get_status():
    """API endpoint to get current system status."""
    global detector
    
    return jsonify({
        'status': 'ready',
        'analysis_available': detector is not None and detector.results is not None,
        'timestamp': datetime.now().isoformat(),
        'symbols_configured': CONFIG['data']['symbols']
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('index.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return render_template('index.html', error="Internal server error"), 500

if __name__ == '__main__':
    app.run(
        host=CONFIG['app']['host'],
        port=CONFIG['app']['port'],
        debug=CONFIG['app']['debug']
    )
