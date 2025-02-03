import torch
import numpy as np
from collections import deque
import asyncio
import websockets
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler

# Import from your existing files
from CNN import CNN  # Your CNN class and functions
from AlbertSeq import CNNALBERT  # Your CNNALBERT class

import os
from pathlib import Path

# Get the absolute path to your project root
PROJECT_ROOT = Path(__file__).parent.parent  # Assuming script is in CTTS folder

# Set the model path using absolute path
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "financial_model_20250202_163730.pth")

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv('COINAPIKEY')
WEBSOCKET_URL = "wss://ws.coinapi.io/v1/"
MODEL_PATH = r"C:\Users\cinco\Desktop\Repos\CTTS_CincoData\CTTS\models\financial_model_20250202_163730.pth"  # Use your actual model file

class LiveMarketPredictor:
    def __init__(self, model_path, sequence_length=80, flat_threshold=0.0001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.flat_threshold = flat_threshold
        self.scaler = MinMaxScaler()
        self.price_buffer = deque(maxlen=sequence_length)
        
        # Load model
        self.model = CNNALBERT().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded with standard accuracy: {checkpoint['standard_accuracy']:.2f}%")
        print(f"Thresholded accuracy: {checkpoint['thresholded_accuracy']:.2f}%")
    
    def preprocess_sequence(self, sequence):
        """Preprocess the price sequence for model input"""
        if len(sequence) < self.sequence_length:
            return None
            
        # Scale the sequence
        scaled_data = self.scaler.fit_transform(np.array(sequence).reshape(-1, 1))
        
        # Reshape for CNN (batch_size, channels, sequence_length)
        sequence_tensor = torch.FloatTensor(scaled_data).view(1, 1, -1).to(self.device)
        return sequence_tensor
    
    def predict(self, sequence_tensor):
        """Make prediction using the model"""
        with torch.no_grad():
            outputs = self.model(sequence_tensor)
            probs = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probs, dim=1)
            confidence = torch.max(probs, dim=1)[0]
            
            pred_map = {0: "UP", 1: "DOWN", 2: "FLAT"}
            all_probs = {
                "UP": probs[0][0].item(),
                "DOWN": probs[0][1].item(),
                "FLAT": probs[0][2].item()
            }
            
            return pred_map[prediction.item()], confidence.item(), all_probs

class MarketDataStream:
    def __init__(self, predictor):
        self.predictor = predictor
        self.last_price = None
        self.output_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Create separate price buffers for each asset
        self.btc_price_buffer = deque(maxlen=predictor.sequence_length)
        self.eth_price_buffer = deque(maxlen=predictor.sequence_length)
        
        # Create and write header to CSV file
        with open(self.output_file, 'w') as f:
            f.write("asset,timestamp,price,prediction,confidence,prob_up,prob_down,prob_flat\n")

    async def process_trade(self, price, timestamp, symbol):
        """Process incoming trade data and make predictions"""
        # Determine which asset we're dealing with and use appropriate buffer
        if "BTC_USDT" in symbol:
            price_buffer = self.btc_price_buffer
            asset = "BTC"
        elif "ETH_USDT" in symbol:
            price_buffer = self.eth_price_buffer
            asset = "ETH"
        else:
            return None
            
        price_buffer.append(price)
        
        if len(price_buffer) == self.predictor.sequence_length:
            sequence_tensor = self.predictor.preprocess_sequence(list(price_buffer))
            if sequence_tensor is not None:
                prediction, confidence, probabilities = self.predictor.predict(sequence_tensor)
                
                # Save prediction results to CSV with asset identifier
                with open(self.output_file, 'a') as f:
                    f.write(f"{asset},{timestamp},{price:.2f},{prediction},{confidence:.4f},"
                           f"{probabilities['UP']:.4f},{probabilities['DOWN']:.4f},"
                           f"{probabilities['FLAT']:.4f}\n")
                
                return {
                    'asset': asset,
                    'timestamp': timestamp,
                    'price': price,
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': probabilities
                }
        return None

    async def start_streaming(self):
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
            
            # Send authentication message
            await websocket.send(json.dumps(hello_message))
            print("Connected to CoinAPI WebSocket")
            print("Streaming trade data and making predictions...")
            
            try:
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if data.get('type') == 'trade':
                        price = float(data.get('price'))
                        timestamp = data.get('time')
                        symbol = data.get('symbol_id')
                        
                        prediction_result = await self.process_trade(price, timestamp, symbol)
                        
                        if prediction_result:
                            print("\nPrediction Result:")
                            print(f"Asset: {prediction_result['asset']}")
                            print(f"Time: {prediction_result['timestamp']}")
                            print(f"Price: ${prediction_result['price']:.2f}")
                            print(f"Direction: {prediction_result['prediction']}")
                            print(f"Confidence: {prediction_result['confidence']:.2f}")
                            print("Probabilities:")
                            for direction, prob in prediction_result['probabilities'].items():
                                print(f"  {direction}: {prob:.2f}")
                    
                    # Add debug print to see all messages
                    else:
                        print(f"Received non-trade message: {data}")
                        
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
            except Exception as e:
                print(f"An error occurred: {e}")
                raise  # Re-raise the exception to see the full error traceback

async def main():
    # Verify the path exists before loading
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        
    print(f"Loading model from: {MODEL_PATH}")
    predictor = LiveMarketPredictor(MODEL_PATH)
    
    # Create data stream
    stream = MarketDataStream(predictor)
    
    try:
        await stream.start_streaming()
    except KeyboardInterrupt:
        print("\nStream stopped by user")
if __name__ == "__main__":
    asyncio.run(main())