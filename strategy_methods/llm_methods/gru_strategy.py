"""
GRU-based trading strategy implementation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import Dict, Any, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class GRUModel(nn.Module):
    """GRU neural network for trading signal prediction"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 output_size: int = 3, dropout: float = 0.2):
        super(GRUModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=dropout)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward propagate GRU
        gru_out, _ = self.gru(x, h0)

        # Apply attention mechanism
        # Transpose for attention: (seq_len, batch, hidden_size)
        gru_out_transposed = gru_out.transpose(0, 1)
        attn_output, _ = self.attention(gru_out_transposed, gru_out_transposed, gru_out_transposed)

        # Use the output from the last time step
        out = self.fc(attn_output[-1])

        return out


class GRUStrategy:
    """GRU-based trading strategy with attention mechanism"""

    def __init__(self, sequence_length: int = 20, hidden_size: int = 64,
                 num_layers: int = 2, learning_rate: float = 0.001,
                 epochs: int = 100, batch_size: int = 32):
        """
        Initialize GRU strategy

        Args:
            sequence_length: Length of input sequences
            hidden_size: Number of GRU hidden units
            num_layers: Number of GRU layers
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

        logger.info(f"GRU Strategy initialized - Device: {self.device}")

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare enhanced features for GRU training"""
        features = data.copy()

        # Flexible column detection
        close_col = 'Close' if 'Close' in features.columns else 'close'
        open_col = 'Open' if 'Open' in features.columns else ('open' if 'open' in features.columns else None)
        high_col = 'High' if 'High' in features.columns else ('high' if 'high' in features.columns else None)
        low_col = 'Low' if 'Low' in features.columns else ('low' if 'low' in features.columns else None)
        volume_col = 'Volume' if 'Volume' in features.columns else ('volume' if 'volume' in features.columns else None)

        # Price-based features
        features['returns'] = features[close_col].pct_change()
        features['log_returns'] = np.log(features[close_col] / features[close_col].shift(1))

        # Multi-timeframe technical indicators
        for window in [5, 10, 12, 20, 26, 50]:
            features[f'sma_{window}'] = features[close_col].rolling(window).mean()
            features[f'ema_{window}'] = features[close_col].ewm(span=window).mean()

        # Advanced technical indicators
        features['macd'] = features['ema_12'] - features['ema_26']
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']

        # Volatility indicators
        features['volatility_20'] = features['returns'].rolling(20).std()
        features['volatility_50'] = features['returns'].rolling(50).std()

        # Volume indicators (if available)
        if volume_col in features.columns:
            features['volume_ma'] = features[volume_col].rolling(20).mean()
            features['volume_ratio'] = features[volume_col] / features['volume_ma']
            features['volume_price_trend'] = (features[close_col] - features[close_col].shift(1)) * features[volume_col]

        # Momentum indicators
        features['roc_5'] = (features[close_col] / features[close_col].shift(5) - 1) * 100
        features['roc_10'] = (features[close_col] / features[close_col].shift(10) - 1) * 100

        # Support/Resistance levels (only if high/low columns exist)
        if high_col and low_col:
            features['high_20'] = features[high_col].rolling(20).max()
            features['low_20'] = features[low_col].rolling(20).min()
            features['price_position'] = (features[close_col] - features['low_20']) / (features['high_20'] - features['low_20'])
        else:
            # Use close price range as fallback
            features['high_20'] = features[close_col].rolling(20).max()
            features['low_20'] = features[close_col].rolling(20).min()
            features['price_position'] = (features[close_col] - features['low_20']) / (features['high_20'] - features['low_20'])

        # RSI with different periods
        for period in [14, 21]:
            delta = features[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Stochastic oscillator
        features['stoch_k'] = ((features[close_col] - features['low_20']) /
                              (features['high_20'] - features['low_20']) * 100)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()

        # Bollinger Bands
        bb_window = 20
        features['bb_middle'] = features[close_col].rolling(bb_window).mean()
        bb_std = features[close_col].rolling(bb_window).std()
        features['bb_upper'] = features['bb_middle'] + (2 * bb_std)
        features['bb_lower'] = features['bb_middle'] - (2 * bb_std)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        features['bb_position'] = (features[close_col] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

        return features

    def create_labels(self, data: pd.DataFrame, future_periods: int = 5,
                     threshold: float = 0.002) -> pd.Series:
        """Create more sophisticated trading labels"""
        # Flexible column detection
        close_col = 'Close' if 'Close' in data.columns else 'close'

        # Multi-period forward returns
        future_returns_3 = data[close_col].shift(-3) / data[close_col] - 1
        future_returns_5 = data[close_col].shift(-5) / data[close_col] - 1
        future_returns_10 = data[close_col].shift(-10) / data[close_col] - 1

        # Weighted average of multi-period returns
        combined_returns = (0.5 * future_returns_3 + 0.3 * future_returns_5 + 0.2 * future_returns_10)

        # Create labels with dynamic thresholds based on volatility
        volatility = data[close_col].pct_change().rolling(20).std()
        dynamic_threshold = threshold * (1 + volatility.fillna(0))

        # Create labels: 0=Hold, 1=Buy, 2=Sell
        labels = pd.Series(0, index=data.index)
        labels[combined_returns > dynamic_threshold] = 1   # Buy
        labels[combined_returns < -dynamic_threshold] = 2  # Sell

        return labels

    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for GRU training"""
        X, y = [], []

        for i in range(len(features) - self.sequence_length + 1):
            if not np.isnan(features[i:i+self.sequence_length]).any() and not np.isnan(labels[i+self.sequence_length-1]):
                X.append(features[i:i+self.sequence_length])
                y.append(labels[i+self.sequence_length-1])

        return np.array(X), np.array(y)

    def train_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the GRU model"""
        logger.info("Starting GRU model training...")

        # Prepare features
        features_df = self.prepare_features(data)
        labels = self.create_labels(data)

        # Select feature columns
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

        # Split into train/validation/test
        n_samples = len(X_tensor)
        train_end = int(0.7 * n_samples)
        val_end = int(0.85 * n_samples)

        X_train, X_val, X_test = X_tensor[:train_end], X_tensor[train_end:val_end], X_tensor[val_end:]
        y_train, y_val, y_test = y_tensor[:train_end], y_tensor[train_end:val_end], y_tensor[val_end:]

        # Initialize model
        input_size = X.shape[2]
        self.model = GRUModel(input_size, self.hidden_size, self.num_layers).to(self.device)

        # Loss and optimizer with weight decay
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 15

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

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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

            # Update learning rate
            scheduler.step(val_loss)

            # Save training history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_gru_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                           f"Val Acc: {val_accuracy:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Load best model and evaluate
        self.model.load_state_dict(torch.load('best_gru_model.pth'))
        self.is_trained = True

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            # Test accuracy
            test_outputs = self.model(X_test)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_accuracy = (test_predicted == y_test).float().mean().item()

            # Validation accuracy
            val_outputs = self.model(X_val)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_accuracy = (val_predicted == y_val).float().mean().item()

        training_results = {
            'success': True,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'best_val_loss': best_val_loss,
            'total_epochs': len(self.training_history),
            'feature_count': input_size,
            'sequence_count': len(X)
        }

        logger.info(f"GRU training completed - Val Acc: {val_accuracy:.3f}, Test Acc: {test_accuracy:.3f}")
        return training_results

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals using trained GRU model"""
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained. Training now...")
            self.train_model(data)

        logger.info("Generating GRU trading signals...")

        # Prepare features
        features_df = self.prepare_features(data)
        feature_cols = [col for col in features_df.columns
                       if col not in ['Close', 'Open', 'High', 'Low', 'Volume', 'close', 'open', 'high', 'low', 'volume']
                       and features_df[col].dtype in ['float64', 'int64']]

        features_array = features_df[feature_cols].fillna(0).values
        features_scaled = self.scaler.transform(features_array)

        # Generate signals
        signals = pd.Series(0, index=data.index)
        confidence = pd.Series(0.0, index=data.index)
        signal_strength = pd.Series(0.0, index=data.index)

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

                    # Calculate signal strength
                    prob_buy = probabilities[0, 1].item()
                    prob_sell = probabilities[0, 2].item()
                    strength = abs(prob_buy - prob_sell)

                    # Convert to trading signal
                    if predicted_class == 1 and max_prob > 0.4:  # Buy with confidence threshold
                        signals.iloc[i] = 1
                        signal_strength.iloc[i] = strength
                    elif predicted_class == 2 and max_prob > 0.4:  # Sell with confidence threshold
                        signals.iloc[i] = -1
                        signal_strength.iloc[i] = -strength
                    else:  # Hold
                        signals.iloc[i] = 0

                    confidence.iloc[i] = max_prob

        # Calculate performance metrics
        labels_actual = self.create_labels(data)
        valid_indices = signals.index.intersection(labels_actual.index)

        if len(valid_indices) > 0:
            signals_binary = (signals[valid_indices] != 0).astype(int)
            labels_binary = (labels_actual[valid_indices] != 0).astype(int)
            hit_rate = accuracy_score(labels_binary, signals_binary) * 100
        else:
            hit_rate = 0

        results = {
            'signals': signals,
            'confidence': confidence,
            'signal_strength': signal_strength,
            'hit_rate': hit_rate,
            'signal_distribution': {
                'buy': (signals == 1).sum(),
                'sell': (signals == -1).sum(),
                'hold': (signals == 0).sum()
            },
            'mean_confidence': confidence[confidence > 0].mean(),
            'strategy_name': 'GRU',
            'model_type': 'Deep Learning'
        }

        logger.info(f"GRU signals generated - Hit Rate: {hit_rate:.1f}%, "
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
            'r2_score': training_results.get('test_accuracy', 0),
            'sharpe_ratio': 0,  # Would need price data to calculate
            'confidence': signal_results.get('mean_confidence', 0),
            'strategy_name': 'GRU',
            'success': True
        }