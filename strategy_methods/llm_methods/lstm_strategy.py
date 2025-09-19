"""
LSTM-based trading strategy implementation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Any, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """LSTM neural network for trading signal prediction"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 output_size: int = 3, dropout: float = 0.2):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Use the output from the last time step
        out = self.fc(out[:, -1, :])

        return out


class LSTMStrategy:
    """LSTM-based trading strategy"""

    def __init__(self, sequence_length: int = 20, hidden_size: int = 64,
                 num_layers: int = 2, learning_rate: float = 0.001,
                 epochs: int = 100, batch_size: int = 32):
        """
        Initialize LSTM strategy

        Args:
            sequence_length: Length of input sequences
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training state
        self.is_trained = False
        self.training_history = []

        logger.info(f"LSTM Strategy initialized - Device: {self.device}")

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for LSTM training"""
        features = data.copy()

        # Price-based features
        close_col = 'Close' if 'Close' in features.columns else 'close'
        features['returns'] = features[close_col].pct_change()
        features['log_returns'] = np.log(features[close_col] / features[close_col].shift(1))

        # Technical indicators
        features['sma_5'] = features[close_col].rolling(5).mean()
        features['sma_20'] = features[close_col].rolling(20).mean()
        features['ema_12'] = features[close_col].ewm(span=12).mean()
        features['ema_26'] = features[close_col].ewm(span=26).mean()

        # Volatility
        features['volatility'] = features['returns'].rolling(20).std()

        # Volume indicators
        volume_col = 'Volume' if 'Volume' in features.columns else 'volume'
        if volume_col in features.columns:
            features['volume_ma'] = features[volume_col].rolling(20).mean()
            features['volume_ratio'] = features[volume_col] / features['volume_ma']

        # Price position indicators (only if columns exist)
        high_col = 'High' if 'High' in features.columns else ('high' if 'high' in features.columns else None)
        low_col = 'Low' if 'Low' in features.columns else ('low' if 'low' in features.columns else None)
        open_col = 'Open' if 'Open' in features.columns else ('open' if 'open' in features.columns else None)

        if high_col and low_col:
            features['high_low_ratio'] = features[high_col] / features[low_col]
        if close_col != open_col and open_col:  # Make sure we have different columns
            features['close_open_ratio'] = features[close_col] / features[open_col]

        # RSI
        delta = features[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        features['macd'] = features['ema_12'] - features['ema_26']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()

        # Bollinger Bands
        features['bb_middle'] = features[close_col].rolling(20).mean()
        bb_std = features[close_col].rolling(20).std()
        features['bb_upper'] = features['bb_middle'] + (2 * bb_std)
        features['bb_lower'] = features['bb_middle'] - (2 * bb_std)
        features['bb_position'] = (features[close_col] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

        return features

    def create_labels(self, data: pd.DataFrame, future_periods: int = 5,
                     threshold: float = 0.002) -> pd.Series:
        """Create trading labels based on future price movements"""
        close_col = 'Close' if 'Close' in data.columns else 'close'
        future_returns = data[close_col].shift(-future_periods) / data[close_col] - 1

        # Create labels: 0=Hold, 1=Buy, 2=Sell
        labels = pd.Series(0, index=data.index)  # Default: Hold
        labels[future_returns > threshold] = 1   # Buy
        labels[future_returns < -threshold] = 2  # Sell

        return labels

    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []

        for i in range(len(features) - self.sequence_length + 1):
            if not np.isnan(features[i:i+self.sequence_length]).any() and not np.isnan(labels[i+self.sequence_length-1]):
                X.append(features[i:i+self.sequence_length])
                y.append(labels[i+self.sequence_length-1])

        return np.array(X), np.array(y)

    def train_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the LSTM model"""
        logger.info("Starting LSTM model training...")

        # Prepare features
        features_df = self.prepare_features(data)
        labels = self.create_labels(data)

        # Select feature columns (exclude target and non-numeric columns)
        feature_cols = [col for col in features_df.columns
                       if col not in ['Close', 'Open', 'High', 'Low', 'Volume', 'close', 'open', 'high', 'low', 'volume']
                       and features_df[col].dtype in ['float64', 'int64']]

        features_array = features_df[feature_cols].fillna(0).values
        labels_array = labels.fillna(0).values

        # Normalize features
        features_scaled = self.scaler.fit_transform(features_array)

        # Create sequences
        X, y = self.create_sequences(features_scaled, labels_array)

        if len(X) == 0:
            logger.error("No valid sequences created. Check data quality.")
            return {'success': False, 'error': 'No valid sequences'}

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # Split into train/validation
        split_idx = int(0.8 * len(X_tensor))
        X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
        y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

        # Initialize model
        input_size = X.shape[2]
        self.model = LSTMModel(input_size, self.hidden_size, self.num_layers).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            num_batches = 0

            for i in range(0, len(X_train), self.batch_size):
                batch_end = min(i + self.batch_size, len(X_train))
                batch_X = X_train[i:batch_end]
                batch_y = y_train[i:batch_end]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss /= num_batches

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()

                # Calculate accuracy
                _, predicted = torch.max(val_outputs.data, 1)
                val_accuracy = (predicted == y_val).float().mean().item()

            # Save training history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_lstm_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Load best model
        self.model.load_state_dict(torch.load('best_lstm_model.pth'))
        self.is_trained = True

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            train_outputs = self.model(X_train)
            _, train_predicted = torch.max(train_outputs.data, 1)
            train_accuracy = (train_predicted == y_train).float().mean().item()

            val_outputs = self.model(X_val)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_accuracy = (val_predicted == y_val).float().mean().item()

        training_results = {
            'success': True,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'best_val_loss': best_val_loss,
            'total_epochs': len(self.training_history),
            'feature_count': input_size,
            'sequence_count': len(X)
        }

        logger.info(f"LSTM training completed - Train Acc: {train_accuracy:.3f}, Val Acc: {val_accuracy:.3f}")
        return training_results

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals using trained LSTM model"""
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained. Training now...")
            self.train_model(data)

        logger.info("Generating LSTM trading signals...")

        # Prepare features - use exact same logic as training
        features_df = self.prepare_features(data)
        feature_cols = [col for col in features_df.columns
                       if col not in ['Close', 'Open', 'High', 'Low', 'Volume', 'close', 'open', 'high', 'low', 'volume']
                       and features_df[col].dtype in ['float64', 'int64']]

        features_array = features_df[feature_cols].fillna(0).values

        # Ensure same number of features as training
        if hasattr(self.scaler, 'n_features_in_'):
            expected_features = self.scaler.n_features_in_
            current_features = features_array.shape[1]

            if current_features != expected_features:
                logger.warning(f"Feature mismatch: expected {expected_features}, got {current_features}")
                # Adjust features to match training
                if current_features < expected_features:
                    # Pad with zeros
                    padding = np.zeros((features_array.shape[0], expected_features - current_features))
                    features_array = np.concatenate([features_array, padding], axis=1)
                else:
                    # Truncate to match
                    features_array = features_array[:, :expected_features]

        features_scaled = self.scaler.transform(features_array)

        # Generate signals
        signals = pd.Series(0, index=data.index)
        confidence = pd.Series(0.0, index=data.index)

        self.model.eval()
        with torch.no_grad():
            for i in range(self.sequence_length, len(features_scaled)):
                # Create sequence
                sequence = features_scaled[i-self.sequence_length:i]

                if not np.isnan(sequence).any():
                    # Predict
                    X_input = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                    output = self.model(X_input)
                    probabilities = torch.softmax(output, dim=1)

                    # Get prediction and confidence
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    max_prob = torch.max(probabilities).item()

                    # Convert to trading signal
                    if predicted_class == 1:  # Buy
                        signals.iloc[i] = 1
                    elif predicted_class == 2:  # Sell
                        signals.iloc[i] = -1
                    else:  # Hold
                        signals.iloc[i] = 0

                    confidence.iloc[i] = max_prob

        # Calculate performance metrics
        labels_actual = self.create_labels(data)
        valid_indices = signals.index.intersection(labels_actual.index)

        if len(valid_indices) > 0:
            # Convert to binary classification for metrics
            signals_binary = (signals[valid_indices] != 0).astype(int)
            labels_binary = (labels_actual[valid_indices] != 0).astype(int)

            hit_rate = accuracy_score(labels_binary, signals_binary) * 100
        else:
            hit_rate = 0

        results = {
            'signals': signals,
            'confidence': confidence,
            'hit_rate': hit_rate,
            'signal_distribution': {
                'buy': (signals == 1).sum(),
                'sell': (signals == -1).sum(),
                'hold': (signals == 0).sum()
            },
            'mean_confidence': confidence[confidence > 0].mean(),
            'strategy_name': 'LSTM',
            'model_type': 'Deep Learning'
        }

        logger.info(f"LSTM signals generated - Hit Rate: {hit_rate:.1f}%, "
                   f"Buy: {results['signal_distribution']['buy']}, "
                   f"Sell: {results['signal_distribution']['sell']}")

        return results

    def train_and_predict(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """
        Interface method for comparison framework
        Train model and generate predictions
        """
        # Create synthetic DataFrame for training
        data = features.copy()
        data['close'] = target

        # Train the model
        training_results = self.train_model(data)
        if not training_results['success']:
            return training_results

        # Generate signals
        signal_results = self.generate_signals(data)

        # Return combined results
        return {
            'signals': signal_results['signals'],
            'hit_rate': signal_results.get('hit_rate', 0) / 100,  # Convert to decimal
            'r2_score': training_results.get('val_accuracy', 0),
            'sharpe_ratio': 0,  # Would need price data to calculate
            'confidence': signal_results.get('mean_confidence', 0),
            'strategy_name': 'LSTM',
            'success': True
        }