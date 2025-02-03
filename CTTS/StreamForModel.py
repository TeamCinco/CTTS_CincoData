import asyncio
import websockets
import json
import os
from dotenv import load_dotenv
import numpy as np
from collections import deque
from datetime import datetime

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv('COINAPIKEY')
WEBSOCKET_URL = "wss://ws.coinapi.io/v1/"

class CryptoDataStream:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.price_buffer = deque(maxlen=sequence_length)
        self.websocket = None
        self.latest_price = None
        self.latest_time = None
        self.latest_symbol = None
        
    async def initialize_connection(self):
        self.websocket = await websockets.connect(WEBSOCKET_URL)
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
        await self.websocket.send(json.dumps(hello_message))
        print("Connected to CoinAPI WebSocket")

    async def process_message(self):
        if self.websocket:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if data.get('type') == 'trade':
                    self.latest_price = float(data.get('price'))
                    self.latest_time = data.get('time')
                    self.latest_symbol = data.get('symbol_id')
                    
                    # Add price to buffer
                    self.price_buffer.append(self.latest_price)
                    
                    return {
                        'price': self.latest_price,
                        'time': self.latest_time,
                        'symbol': self.latest_symbol,
                        'buffer_full': len(self.price_buffer) == self.sequence_length
                    }
                
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                return None
            except Exception as e:
                print(f"Error processing message: {e}")
                return None
        return None

    def get_price_sequence(self):
        """Return the current price sequence if buffer is full"""
        if len(self.price_buffer) == self.sequence_length:
            return np.array(self.price_buffer)
        return None

    async def close(self):
        if self.websocket:
            await self.websocket.close()

async def create_data_stream(sequence_length=60):
    """Factory function to create and initialize a data stream"""
    stream = CryptoDataStream(sequence_length)
    await stream.initialize_connection()
    return stream

# Example usage function
async def demo_stream():
    stream = await create_data_stream()
    try:
        while True:
            data = await stream.process_message()
            if data:
                print(f"\nTrade detected:")
                print(f"Symbol: {data['symbol']}")
                print(f"Price: {data['price']}")
                print(f"Time: {data['time']}")
                if data['buffer_full']:
                    sequence = stream.get_price_sequence()
                    print(f"Sequence ready for prediction, length: {len(sequence)}")
    except KeyboardInterrupt:
        await stream.close()

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(demo_stream())