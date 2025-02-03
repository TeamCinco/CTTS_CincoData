import asyncio
import websockets
import json
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv('COINAPIKEY')

# WebSocket endpoint for CoinAPI
WEBSOCKET_URL = "wss://ws.coinapi.io/v1/"

async def subscribe_to_trades():
    async with websockets.connect(WEBSOCKET_URL) as websocket:
        # Authentication message
        hello_message = {
            "type": "hello",
            "apikey": API_KEY,
            "heartbeat": False,
            "subscribe_data_type": ["trade"],
            "subscribe_filter_symbol_id": [
                "BINANCE_SPOT_BTC_USDT",
                "BINANCE_SPOT_ETH_USDT"
            ]
        }
        
        # Send authentication and subscription message
        await websocket.send(json.dumps(hello_message))
        
        print("Connected to CoinAPI WebSocket")
        print("Streaming trade data...")
        
        # Continuously receive and process messages
        try:
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                # Pretty print the trade data
                if data.get('type') == 'trade':
                    print(f"\nTrade detected:")
                    print(f"Symbol: {data.get('symbol_id')}")
                    print(f"Price: {data.get('price')}")
                    print(f"Size: {data.get('size')}")
                    print(f"Time: {data.get('time')}")
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    try:
        # Install required packages if not already installed
        import pkg_resources
        required_packages = ['websockets', 'python-dotenv']
        installed_packages = [pkg.key for pkg in pkg_resources.working_set]
        
        for package in required_packages:
            if package not in installed_packages:
                print(f"Installing {package}...")
                import subprocess
                subprocess.check_call(['pip', 'install', package])
        
        # Run the WebSocket client
        asyncio.get_event_loop().run_until_complete(subscribe_to_trades())
    except KeyboardInterrupt:
        print("\nStream stopped by user")