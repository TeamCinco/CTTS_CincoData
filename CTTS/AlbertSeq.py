import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from transformers import AlbertConfig, AlbertForSequenceClassification
from CNN import (
    CNN, 
    load_and_process_data, 
    create_sequences, 
    StockDataset, 
    prepare_data
)

import os
import torch
import joblib
from datetime import datetime

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Folder path for data (consistent with previous implementation)
FOLDER_PATH = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\data bento data\SPY data"
#FOLDER_PATH = r"C:\Users\cinco\Desktop\DATA FOR SCRIPTS\data bento data\NVDA"

class CNNALBERT(nn.Module):
    def __init__(self, cnn_embed_dim=128, num_classes=3):
        super(CNNALBERT, self).__init__()
        
        # CNN Component
        self.cnn = CNN(input_channels=1, kernel_size=16, stride=4, embed_dim=cnn_embed_dim)
        
        # ALBERT Configuration
        self.albert_config = AlbertConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_hidden_layers=4,
            intermediate_size=512,
            hidden_dropout_prob=0.3,
            num_labels=num_classes
        )
        
        # ALBERT Component
        self.albert = AlbertForSequenceClassification(self.albert_config)
        
        # Final classification layer
        self.classifier = nn.Linear(self.albert_config.hidden_size, num_classes)
    
    def forward(self, x):
        # Pass through CNN
        cnn_output = self.cnn(x)
        
        # Reshape CNN output for ALBERT
        # ALBERT expects [batch_size, sequence_length, hidden_size]
        batch_size = cnn_output.size(0)
        sequence_length = cnn_output.size(1)
        cnn_output = cnn_output.view(batch_size, sequence_length, -1)
        
        # Pass through ALBERT
        albert_outputs = self.albert(inputs_embeds=cnn_output)
        
        return albert_outputs.logits
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def detailed_model_analysis(model, test_loader, data_loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    # Collect predictions
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Detailed Analysis
    class_names = ['Up', 'Down', 'Flat']
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_targets, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix of Price Movement Predictions')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    plt.close()
    
    # Detailed Classification Report
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    
    # Probability Distribution
    plt.figure(figsize=(12, 6))
    for i in range(3):
        plt.hist(
            all_probs[all_targets == i][:, i], 
            bins=50, 
            alpha=0.5, 
            label=class_names[i]
        )
    plt.title('Probability Distribution by True Class')
    plt.xlabel('Prediction Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/probability_distribution.png')
    plt.close()
    
    # High Confidence Prediction Analysis
    thresholds = [0.6, 0.7, 0.8, 0.9]
    print("\nHigh Confidence Prediction Analysis:")
    for threshold in thresholds:
        high_conf_mask = np.max(all_probs, axis=1) >= threshold
        high_conf_preds = all_preds[high_conf_mask]
        high_conf_targets = all_targets[high_conf_mask]
        
        accuracy = np.mean(high_conf_preds == high_conf_targets)
        print(f"Threshold {threshold}: Accuracy = {accuracy*100:.2f}%, Samples = {len(high_conf_preds)}")
    
    # Create a comprehensive DataFrame
    results_df = pd.DataFrame({
        'True_Label': all_targets,
        'Predicted_Label': all_preds,
        'Prob_Up': all_probs[:, 0],
        'Prob_Down': all_probs[:, 1],
        'Prob_Flat': all_probs[:, 2],
        'Max_Probability': np.max(all_probs, axis=1)
    })
    results_df['Correct_Prediction'] = results_df['True_Label'] == results_df['Predicted_Label']
    
    # Save results to CSV
    results_df.to_csv('models/prediction_results.csv', index=False)
    
    # Time Series Visualization
    plt.figure(figsize=(15, 7))
    
    # Time Series Visualization
    actual_prices = test_loader.dataset.original_prices
    timestamps = test_loader.dataset.timestamps
    
    # Plot actual price movement
    plt.plot(timestamps, actual_prices, label='Actual Price', color='black', alpha=0.5)
    
    # Color-code predictions
    colors = {0: 'green', 1: 'red', 2: 'gray'}  # Up, Down, Flat
    markers = {0: '^', 1: 'v', 2: 'o'}
    
    for i, (pred, target) in enumerate(zip(all_preds, all_targets)):
        color = colors[pred]
        marker = markers[pred]
        plt.scatter(timestamps[i], actual_prices[i], 
                    color=color, 
                    marker=marker, 
                    s=100, 
                    alpha=0.7, 
                    edgecolors='black',
                    linewidth=1.5)
    
    plt.title('Price Movement with Model Predictions')
    plt.xlabel('Timestamp')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('models/time_series_predictions.png')
    plt.close()
    
    return results_df
def train_combined_model(model, train_loader, num_epochs=100, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        if (epoch + 1) % 10 == 0:
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

def evaluate_combined_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probs, 1)
            
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
            all_probs.extend(max_probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    standard_accuracy = 100 * correct / total
    
    # Calculate thresholded accuracy
    threshold = np.percentile(all_probs, 75)
    high_conf_indices = np.where(np.array(all_probs) >= threshold)[0]
    
    thresholded_correct = sum(np.array(all_preds)[high_conf_indices] == np.array(all_targets)[high_conf_indices])
    thresholded_accuracy = 100 * thresholded_correct / len(high_conf_indices)
    
    return standard_accuracy, thresholded_accuracy

def main():
    print(f"Using device: {device}")
    
    # Create a timestamp for unique model naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare data using the imported prepare_data function and predefined folder path
    train_loader, test_loader = prepare_data(FOLDER_PATH)
    
    # Initialize combined model
    model = CNNALBERT().to(device)
    
    # Train the model
    print("\nStarting training...")
    train_combined_model(model, train_loader)
    
    # Evaluate the model
    print("\nEvaluating model...")
    standard_acc, thresholded_acc = evaluate_combined_model(model, test_loader)
    print(f'Final Standard Test Accuracy: {standard_acc:.2f}%')
    print(f'Final Thresholded Test Accuracy: {thresholded_acc:.2f}%')
    
    # Perform detailed analysis
    results_df = detailed_model_analysis(model, test_loader)
    # Create a models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model with timestamp
    model_filename = f'models/financial_model_{timestamp}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'standard_accuracy': standard_acc,
        'thresholded_accuracy': thresholded_acc,
        'timestamp': timestamp
    }, model_filename)
    print(f"Model saved to {model_filename}")
    
    # Save model performance log
    performance_log = f'models/performance_log_{timestamp}.txt'
    with open(performance_log, 'w') as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Standard Accuracy: {standard_acc:.2f}%\n")
        f.write(f"Thresholded Accuracy: {thresholded_acc:.2f}%\n")
    print(f"Performance log saved to {performance_log}")

if __name__ == "__main__":
    main()