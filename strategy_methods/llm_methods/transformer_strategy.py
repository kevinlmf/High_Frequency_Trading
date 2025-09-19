"""
Transformer-based trading strategy implementation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import Dict, Any, Tuple, Optional
import math
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    """Transformer neural network for trading signal prediction"""

    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8,
                 num_encoder_layers: int = 4, dim_feedforward: int = 512,
                 dropout: float = 0.1, output_size: int = 3):
        super(TransformerModel, self).__init__()

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_size)
        )

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize model weights"""
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()

    def forward(self, x):
        # Input projection
        x = self.input_projection(x) * math.sqrt(self.d_model)

        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Global pooling
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, d_model)

        # Classification
        output = self.classifier(x)

        return output


class TransformerStrategy:
    """Transformer-based trading strategy"""

    def __init__(self, sequence_length: int = 30, d_model: int = 128, nhead: int = 8,
                 num_encoder_layers: int = 4, learning_rate: float = 0.0001,
                 epochs: int = 150, batch_size: int = 32, warmup_steps: int = 1000):
        """
        Initialize Transformer strategy

        Args:
            sequence_length: Length of input sequences
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Batch size for training
            warmup_steps: Number of warmup steps for learning rate scheduler
        """
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps

        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training state
        self.is_trained = False
        self.training_history = []

        logger.info(f"Transformer Strategy initialized - Device: {self.device}")

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive features for Transformer training"""
        features = data.copy()

        # Flexible column detection
        close_col = 'Close' if 'Close' in features.columns else 'close'
        open_col = 'Open' if 'Open' in features.columns else 'open'
        high_col = 'High' if 'High' in features.columns else 'high'
        low_col = 'Low' if 'Low' in features.columns else 'low'
        volume_col = 'Volume' if 'Volume' in features.columns else 'volume'

        # Price-based features
        features['returns'] = features[close_col].pct_change()
        features['log_returns'] = np.log(features[close_col] / features[close_col].shift(1))

        # Multi-timeframe moving averages
        for window in [3, 5, 8, 13, 21, 34, 55]:
            features[f'sma_{window}'] = features[close_col].rolling(window).mean()
            features[f'ema_{window}'] = features[close_col].ewm(span=window).mean()
            features[f'price_to_sma_{window}'] = features[close_col] / features[f'sma_{window}']

        # Advanced technical indicators
        # MACD family
        ema_12 = features[close_col].ewm(span=12).mean()
        ema_26 = features[close_col].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']

        # Volatility measures
        for window in [10, 20, 30]:
            features[f'volatility_{window}'] = features['returns'].rolling(window).std()
            features[f'volatility_ratio_{window}'] = features[f'volatility_{window}'] / features['volatility_20']

        # Volume indicators (if available)
        if volume_col in features.columns:
            for window in [5, 10, 20]:
                features[f'volume_ma_{window}'] = features[volume_col].rolling(window).mean()
                features[f'volume_ratio_{window}'] = features[volume_col] / features[f'volume_ma_{window}']

            features['volume_price_trend'] = (features[close_col] - features[close_col].shift(1)) * features[volume_col]
            features['on_balance_volume'] = (features['returns'] > 0).astype(int) * features[volume_col] - \
                                          (features['returns'] < 0).astype(int) * features[volume_col]
            features['on_balance_volume'] = features['on_balance_volume'].cumsum()

        # Momentum indicators
        for period in [3, 5, 10, 20]:
            features[f'roc_{period}'] = (features[close_col] / features[close_col].shift(period) - 1) * 100

        # RSI with multiple periods
        for period in [7, 14, 21]:
            delta = features[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Stochastic oscillators
        for k_period in [14, 21]:
            for d_period in [3, 5]:
                high_k = features[high_col].rolling(k_period).max()
                low_k = features[low_col].rolling(k_period).min()
                features[f'stoch_k_{k_period}'] = ((features[close_col] - low_k) / (high_k - low_k)) * 100
                features[f'stoch_d_{k_period}_{d_period}'] = features[f'stoch_k_{k_period}'].rolling(d_period).mean()

        # Bollinger Bands
        for window in [20, 30]:
            bb_middle = features[close_col].rolling(window).mean()
            bb_std = features[close_col].rolling(window).std()
            features[f'bb_upper_{window}'] = bb_middle + (2 * bb_std)
            features[f'bb_lower_{window}'] = bb_middle - (2 * bb_std)
            features[f'bb_width_{window}'] = (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}']) / bb_middle
            features[f'bb_position_{window}'] = (features[close_col] - features[f'bb_lower_{window}']) / \
                                               (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}'])

        # Support/Resistance levels
        for window in [10, 20, 50]:
            features[f'high_{window}'] = features[high_col].rolling(window).max()
            features[f'low_{window}'] = features[low_col].rolling(window).min()
            features[f'price_position_{window}'] = (features[close_col] - features[f'low_{window}']) / \
                                                   (features[f'high_{window}'] - features[f'low_{window}'])

        # Candlestick patterns (simplified)
        features['body_size'] = abs(features[close_col] - features[open_col])
        features['upper_shadow'] = features[high_col] - np.maximum(features[close_col], features[open_col])
        features['lower_shadow'] = np.minimum(features[close_col], features[open_col]) - features[low_col]
        features['total_range'] = features[high_col] - features[low_col]

        # Normalized price position
        features['hl2'] = (features[high_col] + features[low_col]) / 2
        features['hlc3'] = (features[high_col] + features[low_col] + features[close_col]) / 3
        features['ohlc4'] = (features[open_col] + features[high_col] + features[low_col] + features[close_col]) / 4

        return features

    def create_labels(self, data: pd.DataFrame, future_periods: int = 5,
                     threshold: float = 0.002) -> pd.Series:
        """Create sophisticated multi-horizon labels"""
        # Flexible column detection
        close_col = 'Close' if 'Close' in data.columns else 'close'

        # Multi-horizon returns
        returns_1 = data[close_col].shift(-1) / data[close_col] - 1
        returns_3 = data[close_col].shift(-3) / data[close_col] - 1
        returns_5 = data[close_col].shift(-5) / data[close_col] - 1
        returns_10 = data[close_col].shift(-10) / data[close_col] - 1

        # Weighted multi-horizon returns
        combined_returns = (0.4 * returns_1 + 0.3 * returns_3 + 0.2 * returns_5 + 0.1 * returns_10)

        # Adaptive threshold based on volatility
        volatility = data[close_col].pct_change().rolling(30).std()
        adaptive_threshold = threshold * (1 + 2 * volatility.fillna(volatility.mean()))

        # Create labels with trend strength consideration
        labels = pd.Series(0, index=data.index)

        # Strong buy/sell signals
        strong_buy = combined_returns > 2 * adaptive_threshold
        strong_sell = combined_returns < -2 * adaptive_threshold

        # Regular buy/sell signals
        buy = (combined_returns > adaptive_threshold) & ~strong_buy
        sell = (combined_returns < -adaptive_threshold) & ~strong_sell

        labels[strong_buy] = 1   # Strong Buy
        labels[buy] = 1          # Buy
        labels[strong_sell] = 2  # Strong Sell
        labels[sell] = 2         # Sell

        return labels

    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for Transformer training"""
        X, y = [], []

        for i in range(len(features) - self.sequence_length + 1):
            if not np.isnan(features[i:i+self.sequence_length]).any() and not np.isnan(labels[i+self.sequence_length-1]):
                X.append(features[i:i+self.sequence_length])
                y.append(labels[i+self.sequence_length-1])

        return np.array(X), np.array(y)

    def get_lr_schedule(self, step: int) -> float:
        """Learning rate schedule with warmup"""
        if step < self.warmup_steps:
            return self.learning_rate * step / self.warmup_steps
        else:
            return self.learning_rate * 0.5 ** ((step - self.warmup_steps) / (self.epochs * 0.1))

    def train_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the Transformer model"""
        logger.info("Starting Transformer model training...")

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

        # Split data
        n_samples = len(X_tensor)
        train_end = int(0.7 * n_samples)
        val_end = int(0.85 * n_samples)

        X_train, X_val, X_test = X_tensor[:train_end], X_tensor[train_end:val_end], X_tensor[val_end:]
        y_train, y_val, y_test = y_tensor[:train_end], y_tensor[train_end:val_end], y_tensor[val_end:]

        # Initialize model
        input_size = X.shape[2]
        self.model = TransformerModel(
            input_size=input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                               weight_decay=1e-4, betas=(0.9, 0.98), eps=1e-9)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        global_step = 0

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            num_batches = 0

            # Shuffle training data
            indices = torch.randperm(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range(0, len(X_train_shuffled), self.batch_size):
                batch_end = min(i + self.batch_size, len(X_train_shuffled))
                batch_X = X_train_shuffled[i:batch_end]
                batch_y = y_train_shuffled[i:batch_end]

                # Update learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.get_lr_schedule(global_step)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                optimizer.step()
                global_step += 1

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
                'val_accuracy': val_accuracy,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_transformer_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 25 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                           f"Val Acc: {val_accuracy:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Load best model and evaluate
        self.model.load_state_dict(torch.load('best_transformer_model.pth'))
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
            'sequence_count': len(X),
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }

        logger.info(f"Transformer training completed - Val Acc: {val_accuracy:.3f}, Test Acc: {test_accuracy:.3f}")
        return training_results

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals using trained Transformer model"""
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained. Training now...")
            self.train_model(data)

        logger.info("Generating Transformer trading signals...")

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
                    prob_hold = probabilities[0, 0].item()

                    # Enhanced signal generation with confidence thresholds
                    confidence_threshold = 0.45
                    strong_confidence_threshold = 0.6

                    if predicted_class == 1 and max_prob > confidence_threshold:  # Buy
                        if max_prob > strong_confidence_threshold:
                            signals.iloc[i] = 1
                            signal_strength.iloc[i] = prob_buy - prob_hold
                        else:
                            signals.iloc[i] = 1
                            signal_strength.iloc[i] = (prob_buy - prob_hold) * 0.7
                    elif predicted_class == 2 and max_prob > confidence_threshold:  # Sell
                        if max_prob > strong_confidence_threshold:
                            signals.iloc[i] = -1
                            signal_strength.iloc[i] = -(prob_sell - prob_hold)
                        else:
                            signals.iloc[i] = -1
                            signal_strength.iloc[i] = -(prob_sell - prob_hold) * 0.7
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
            'strategy_name': 'Transformer',
            'model_type': 'Deep Learning'
        }

        logger.info(f"Transformer signals generated - Hit Rate: {hit_rate:.1f}%, "
                   f"Buy: {results['signal_distribution']['buy']}, "
                   f"Sell: {results['signal_distribution']['sell']}")

        return results