#!/usr/bin/env python3
"""
Interactive Insider Trading Detection System
Run this script to start analyzing stocks for insider trading patterns
"""

import os
import time
from simple_analyzer import SimpleInsiderTradingDetector
from datetime import datetime

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the application header"""
    print("=" * 80)
    print("🚨 INSIDER TRADING DETECTION SYSTEM")
    print("=" * 80)
    print(f"📅 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)

def main_menu():
    """Display the main menu and handle user input"""
    detector = SimpleInsiderTradingDetector()
    
    while True:
        clear_screen()
        print_header()
        
        print("📋 MENU OPTIONS:")
        print("1. Quick Analysis (5 major stocks)")
        print("2. Custom Stock Analysis")
        print("3. Recent Activity Check")
        print("4. Full Report")
        print("5. Exit")
        print("-" * 80)
        
        choice = input("Select option (1-5): ").strip()
        
        if choice == '1':
            # Quick analysis
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
            detector.analyze_stocks(symbols, days_back=90)
            detector.show_top_anomalies(10)
            input("\nPress Enter to continue...")
            
        elif choice == '2':
            # Custom analysis
            print("\nEnter stock symbols separated by commas (e.g., AAPL,NVDA,META):")
            symbols_input = input("Symbols: ").strip().upper()
            if symbols_input:
                symbols = [s.strip() for s in symbols_input.split(',')]
                days = input("Days to analyze (default 180): ").strip()
                days = int(days) if days.isdigit() else 180
                
                detector.analyze_stocks(symbols, days_back=days)
                detector.simple_report()
            input("\nPress Enter to continue...")
            
        elif choice == '3':
            # Recent activity
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META']
            detector.analyze_stocks(symbols, days_back=30)
            detector.show_recent_activity(7)
            input("\nPress Enter to continue...")
            
        elif choice == '4':
            # Full report
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'ORCL', 'CRM']
            detector.analyze_stocks(symbols, days_back=365)
            detector.simple_report()
            input("\nPress Enter to continue...")
            
        elif choice == '5':
            print("\nThanks for using the Insider Trading Detection System!")
            break
            
        else:
            print("Invalid option. Please try again.")
            time.sleep(1)

def auto_analysis():
    """Run automatic analysis every hour"""
    detector = SimpleInsiderTradingDetector()
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META']
    
    while True:
        clear_screen()
        print_header()
        print("🔄 AUTOMATIC ANALYSIS MODE")
        print("-" * 80)
        
        try:
            detector.analyze_stocks(symbols, days_back=30)
            detector.show_recent_activity(1)  # Show today's activity
            
            print(f"\n⏰ Next update in 1 hour...")
            print("Press Ctrl+C to stop automatic mode")
            
            # Wait 1 hour (3600 seconds)
            time.sleep(3600)
            
        except KeyboardInterrupt:
            print("\n\nAutomatic analysis stopped.")
            break
        except Exception as e:
            print(f"\nError in automatic analysis: {e}")
            print("Retrying in 5 minutes...")
            time.sleep(300)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        auto_analysis()
    else:
        main_menu()