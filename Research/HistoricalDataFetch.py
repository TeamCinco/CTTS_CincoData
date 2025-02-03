import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

class CoinAPIHistoricalData:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('COINAPIKEY')
        self.base_url = "https://rest.coinapi.io/v1"
        self.headers = {'X-CoinAPI-Key': self.api_key}
    
    def get_available_symbols(self):
        """Fetch available symbols from the API"""
        endpoint = f"{self.base_url}/symbols"
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            # Filter for BINANCE SPOT symbols
            spot_symbols = [
                symbol for symbol in data 
                if symbol.get('exchange_id') == 'BINANCE' 
                and 'SPOT' in symbol.get('symbol_type', '')
            ]
            return spot_symbols
        except requests.exceptions.RequestException as e:
            print(f"Error fetching symbols: {e}")
            return None

    def get_available_periods(self):
        """Fetch available time periods"""
        endpoint = f"{self.base_url}/ohlcv/periods"
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching periods: {e}")
            return None
    
    def fetch_ohlcv(self, symbol, period_id, time_start=None, time_end=None, limit=1000):
        endpoint = f"{self.base_url}/ohlcv/{symbol}/history"
        params = {
            'period_id': period_id,
            'limit': limit
        }
        
        if time_start:
            params['time_start'] = time_start
        if time_end:
            params['time_end'] = time_end
            
        try:
            response = requests.get(endpoint, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            df = pd.DataFrame(data)
            
            raw_filename = f"{symbol}_{period_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(raw_filename, index=False)
            print(f"Data saved to {raw_filename}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None
    def get_available_timeframes(self, symbol, period_id):
        """Fetch available time ranges for the given symbol and period"""
        endpoint = f"{self.base_url}/ohlcv/{symbol}/history/info"
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            # Filter for the selected period
            period_data = [d for d in data if d.get('period_id') == period_id]
            if period_data:
                return {
                    'first_historical_data': period_data[0].get('first_historical_data'),
                    'last_historical_data': period_data[0].get('last_historical_data')
                }
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching time ranges: {e}")
            return None
def main():
    fetcher = CoinAPIHistoricalData()
    
    # Fetch available options
    print("Fetching available trading pairs...")
    symbols = fetcher.get_available_symbols()
    if not symbols:
        print("Could not fetch available symbols. Exiting.")
        return
        
    print("Fetching available time periods...")
    periods = fetcher.get_available_periods()
    if not periods:
        print("Could not fetch available periods. Exiting.")
        return

    # Display available symbols
    print("\nAvailable Trading Pairs:")
    btc_symbols = [s for s in symbols if 'BTC' in s.get('symbol_id', '')]
    for i, symbol in enumerate(btc_symbols, 1):
        print(f"{i}. {symbol['symbol_id']}")
    
    # Get user symbol choice with validation
    while True:
        try:
            symbol_choice = int(input(f"\nSelect trading pair number (1-{len(btc_symbols)}): "))
            if 1 <= symbol_choice <= len(btc_symbols):
                selected_symbol = btc_symbols[symbol_choice-1]['symbol_id']
                break
            print(f"Please enter a number between 1 and {len(btc_symbols)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Display available periods
    print("\nAvailable Time Periods:")
    for i, period in enumerate(periods, 1):
        print(f"{i}. {period['period_id']} ({period['length_seconds']} seconds)")
    
    # Get user period choice with validation
    while True:
        try:
            period_choice = int(input(f"\nSelect period number (1-{len(periods)}): "))
            if 1 <= period_choice <= len(periods):
                selected_period = periods[period_choice-1]['period_id']
                break
            print(f"Please enter a number between 1 and {len(periods)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Set time range to maximum available
    time_end = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Fetch data with maximum limit
    df = fetcher.fetch_ohlcv(
        symbol=selected_symbol,
        period_id=selected_period,
        time_end=time_end,
        limit=100000  # Set to maximum limit allowed by API
    )
    
    if df is not None:
        print("\nData Preview:")
        print(df.head())
        print(f"\nTotal records: {len(df)}")
        print(f"\nDate range in retrieved data:")
        if not df.empty:
            print(f"From: {df['time_period_start'].min()}")
            print(f"To: {df['time_period_end'].max()}")

if __name__ == "__main__":
    main()