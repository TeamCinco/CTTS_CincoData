import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def visualize_trading_data(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert timestamp to datetime if it's not None
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Cryptocurrency Price and Predictions Over Time', fontsize=16)
    
    # Color mapping for predictions
    color_map = {'UP': 'green', 'DOWN': 'red', 'FLAT': 'blue'}
    
    # Plot BTC data
    btc_data = df[df['asset'] == 'BTC']
    if not btc_data.empty:
        ax1.plot(btc_data.index, btc_data['price'], label='BTC Price', color='black', alpha=0.6)
        
        # Add colored points for predictions
        for pred in color_map.keys():
            mask = btc_data['prediction'] == pred
            ax1.scatter(btc_data[mask].index, btc_data[mask]['price'], 
                       color=color_map[pred], label=f'Predict {pred}', alpha=0.6)
        
        ax1.set_title('BTC Price and Predictions')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot ETH data
    eth_data = df[df['asset'] == 'ETH']
    if not eth_data.empty:
        ax2.plot(eth_data.index, eth_data['price'], label='ETH Price', color='black', alpha=0.6)
        
        # Add colored points for predictions
        for pred in color_map.keys():
            mask = eth_data['prediction'] == pred
            ax2.scatter(eth_data[mask].index, eth_data[mask]['price'], 
                       color=color_map[pred], label=f'Predict {pred}', alpha=0.6)
        
        ax2.set_title('ETH Price and Predictions')
        ax2.set_ylabel('Price (USD)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Add confidence visualization
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(15, 10))
    fig2.suptitle('Prediction Probabilities Over Time', fontsize=16)
    
    # Plot BTC probabilities
    if not btc_data.empty:
        ax3.plot(btc_data.index, btc_data['prob_up'], label='UP', color='green', alpha=0.6)
        ax3.plot(btc_data.index, btc_data['prob_down'], label='DOWN', color='red', alpha=0.6)
        ax3.plot(btc_data.index, btc_data['prob_flat'], label='FLAT', color='blue', alpha=0.6)
        ax3.set_title('BTC Prediction Probabilities')
        ax3.set_ylabel('Probability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot ETH probabilities
    if not eth_data.empty:
        ax4.plot(eth_data.index, eth_data['prob_up'], label='UP', color='green', alpha=0.6)
        ax4.plot(eth_data.index, eth_data['prob_down'], label='DOWN', color='red', alpha=0.6)
        ax4.plot(eth_data.index, eth_data['prob_flat'], label='FLAT', color='blue', alpha=0.6)
        ax4.set_title('ETH Prediction Probabilities')
        ax4.set_ylabel('Probability')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Usage
if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file = r"C:\Users\cinco\Desktop\Repos\CTTS_CincoData\predictions_20250202_195011.csv"  # Update this to your CSV filename
    visualize_trading_data(csv_file)