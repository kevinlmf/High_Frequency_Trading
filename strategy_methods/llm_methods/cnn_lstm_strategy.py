"""
CNN-LSTM hybrid trading strategy implementation
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


class CNNLSTMModel(nn.Module):
    """CNN-LSTM hybrid model for trading signal prediction"""

    def __init__(self, input_size: int, cnn_channels: int = 64, lstm_hidden: int = 128,
                 lstm_layers: int = 2, output_size: int = 3, dropout: float = 0.2):
        super(CNNLSTMModel, self).__init__()

        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        # CNN layers for feature extraction
        self.conv1d_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(input_size, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Second conv block
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Third conv block
            nn.Conv1d(cnn_channels * 2, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,  # bidirectional
            num_heads=8,
            dropout=dropout
        )

        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, output_size)
        )

    def forward(self, x):
        batch_size, seq_len, features = x.size()

        # CNN feature extraction
        # Reshape for conv1d: (batch, features, seq_len)
        x = x.transpose(1, 2)
        cnn_out = self.conv1d_layers(x)

        # Reshape back for LSTM: (batch, seq_len, features)
        cnn_out = cnn_out.transpose(1, 2)

        # LSTM processing
        h0 = torch.zeros(self.lstm_layers * 2, batch_size, self.lstm_hidden).to(x.device)
        c0 = torch.zeros(self.lstm_layers * 2, batch_size, self.lstm_hidden).to(x.device)

        lstm_out, _ = self.lstm(cnn_out, (h0, c0))

        # Attention mechanism
        # Transpose for attention: (seq_len, batch, features)
        lstm_out_transposed = lstm_out.transpose(0, 1)
        attn_output, attn_weights = self.attention(
            lstm_out_transposed, lstm_out_transposed, lstm_out_transposed
        )

        # Use the last time step output
        final_output = self.classifier(attn_output[-1])

        return final_output


class CNNLSTMStrategy:
    """CNN-LSTM hybrid trading strategy"""

    def __init__(self, sequence_length: int = 25, cnn_channels: int = 64,
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 learning_rate: float = 0.0005, epochs: int = 120, batch_size: int = 32):
        """
        Initialize CNN-LSTM strategy

        Args:
            sequence_length: Length of input sequences
            cnn_channels: Number of CNN channels
            lstm_hidden: LSTM hidden size
            lstm_layers: Number of LSTM layers
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.sequence_length = sequence_length
        self.cnn_channels = cnn_channels
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
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

        logger.info(f"CNN-LSTM Strategy initialized - Device: {self.device}")

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features optimized for CNN-LSTM architecture"""
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

        # Multi-scale moving averages
        for window in [3, 5, 8, 13, 21, 34, 55, 89]:
            features[f'sma_{window}'] = features[close_col].rolling(window).mean()
            features[f'ema_{window}'] = features[close_col].ewm(span=window).mean()
            features[f'price_to_ma_{window}'] = features[close_col] / features[f'sma_{window}']

        # Technical oscillators
        # RSI with multiple periods
        for period in [7, 14, 21, 30]:
            delta = features[close_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD family with different settings
        for fast, slow, signal in [(8, 21, 5), (12, 26, 9), (19, 39, 9)]:
            ema_fast = features[close_col].ewm(span=fast).mean()
            ema_slow = features[close_col].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            features[f'macd_{fast}_{slow}'] = macd
            features[f'macd_signal_{fast}_{slow}_{signal}'] = macd_signal
            features[f'macd_hist_{fast}_{slow}_{signal}'] = macd - macd_signal

        # Volatility indicators
        for window in [5, 10, 20, 30]:
            features[f'volatility_{window}'] = features['returns'].rolling(window).std()
            features[f'atr_{window}'] = (features[high_col] - features[low_col]).rolling(window).mean()

        # Volume-based indicators (if available)
        if volume_col in features.columns:
            for window in [5, 10, 20]:
                features[f'volume_sma_{window}'] = features[volume_col].rolling(window).mean()
                features[f'volume_ratio_{window}'] = features[volume_col] / features[f'volume_sma_{window}']

            # Volume-price indicators
            features['volume_weighted_price'] = (features[volume_col] * features[close_col]).rolling(20).sum() / \
                                               features[volume_col].rolling(20).sum()
            features['price_volume_trend'] = ((features[close_col] - features[close_col].shift(1)) /
                                             features[close_col].shift(1) * features[volume_col]).cumsum()

        # Momentum and trend indicators
        for period in [5, 10, 20, 30]:
            features[f'roc_{period}'] = (features[close_col] / features[close_col].shift(period) - 1) * 100
            features[f'momentum_{period}'] = features[close_col] - features[close_col].shift(period)

        # Bollinger Bands with multiple settings
        for window, std_mult in [(20, 2), (20, 2.5), (30, 2)]:
            bb_middle = features[close_col].rolling(window).mean()
            bb_std = features[close_col].rolling(window).std()
            features[f'bb_upper_{window}_{std_mult}'] = bb_middle + (std_mult * bb_std)
            features[f'bb_lower_{window}_{std_mult}'] = bb_middle - (std_mult * bb_std)
            features[f'bb_width_{window}_{std_mult}'] = (features[f'bb_upper_{window}_{std_mult}'] -
                                                        features[f'bb_lower_{window}_{std_mult}']) / bb_middle
            features[f'bb_position_{window}_{std_mult}'] = (features[close_col] - features[f'bb_lower_{window}_{std_mult}']) / \
                                                          (features[f'bb_upper_{window}_{std_mult}'] -
                                                           features[f'bb_lower_{window}_{std_mult}'])

        # Stochastic oscillators
        for k_period in [14, 21]:
            high_k = features[high_col].rolling(k_period).max()
            low_k = features[low_col].rolling(k_period).min()
            features[f'stoch_k_{k_period}'] = ((features[close_col] - low_k) / (high_k - low_k)) * 100
            features[f'stoch_d_{k_period}'] = features[f'stoch_k_{k_period}'].rolling(3).mean()

        # Williams %R
        for period in [14, 21]:
            high_period = features[high_col].rolling(period).max()
            low_period = features[low_col].rolling(period).min()
            features[f'williams_r_{period}'] = -100 * (high_period - features[close_col]) / (high_period - low_period)

        # Price channel indicators
        for window in [10, 20, 30]:
            features[f'high_{window}'] = features[high_col].rolling(window).max()
            features[f'low_{window}'] = features[low_col].rolling(window).min()
            features[f'channel_position_{window}'] = (features[close_col] - features[f'low_{window}']) / \
                                                    (features[f'high_{window}'] - features[f'low_{window}'])

        # Candlestick pattern features
        features['body_size'] = abs(features[close_col] - features[open_col])
        features['upper_shadow'] = features[high_col] - np.maximum(features[close_col], features[open_col])
        features['lower_shadow'] = np.minimum(features[close_col], features[open_col]) - features[low_col]
        features['shadow_ratio'] = (features['upper_shadow'] + features['lower_shadow']) / features['body_size']

        # Statistical features
        for window in [10, 20]:
            features[f'skewness_{window}'] = features['returns'].rolling(window).skew()
            features[f'kurtosis_{window}'] = features['returns'].rolling(window).kurt()

        return features

    def create_labels(self, data: pd.DataFrame, future_periods: int = 5,
                     threshold: float = 0.0025) -> pd.Series:
        """Create sophisticated multi-period labels"""
        # Flexible column detection
        close_col = 'Close' if 'Close' in data.columns else 'close'

        # Multiple future horizons
        horizons = [2, 5, 8, 10]
        weights = [0.4, 0.3, 0.2, 0.1]

        weighted_returns = pd.Series(0.0, index=data.index)
        for horizon, weight in zip(horizons, weights):
            future_return = data[close_col].shift(-horizon) / data[close_col] - 1
            weighted_returns += weight * future_return

        # Dynamic threshold based on recent volatility
        volatility = data[close_col].pct_change().rolling(30).std()
        dynamic_threshold = threshold * (1 + volatility.fillna(volatility.mean()))

        # Trend strength consideration
        trend_strength = abs(data[close_col].rolling(20).mean() - data[close_col].rolling(50).mean()) / \
                        data[close_col].rolling(50).mean()

        enhanced_threshold = dynamic_threshold * (1 + trend_strength.fillna(0))

        # Create labels
        labels = pd.Series(0, index=data.index)
        labels[weighted_returns > enhanced_threshold] = 1   # Buy
        labels[weighted_returns < -enhanced_threshold] = 2  # Sell

        return labels

    def create_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for CNN-LSTM training"""
        X, y = [], []

        for i in range(len(features) - self.sequence_length + 1):
            if not np.isnan(features[i:i+self.sequence_length]).any() and not np.isnan(labels[i+self.sequence_length-1]):
                X.append(features[i:i+self.sequence_length])
                y.append(labels[i+self.sequence_length-1])

        return np.array(X), np.array(y)

    def train_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the CNN-LSTM model"""
        logger.info("Starting CNN-LSTM model training...")

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

        # Class weights for imbalanced data
        class_counts = torch.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts.float())
        class_weights = class_weights.to(self.device)

        # Initialize model
        input_size = X.shape[2]
        self.model = CNNLSTMModel(
            input_size=input_size,
            cnn_channels=self.cnn_channels,
            lstm_hidden=self.lstm_hidden,
            lstm_layers=self.lstm_layers
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=8, factor=0.7, verbose=True
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            # Shuffle training data
            indices = torch.randperm(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range(0, len(X_train_shuffled), self.batch_size):
                batch_end = min(i + self.batch_size, len(X_train_shuffled))
                batch_X = X_train_shuffled[i:batch_end]
                batch_y = y_train_shuffled[i:batch_end]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            train_loss /= (len(X_train_shuffled) // self.batch_size)
            train_accuracy = train_correct / train_total

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()

                _, val_predicted = torch.max(val_outputs.data, 1)
                val_accuracy = (val_predicted == y_val).float().mean().item()

            # Update learning rate
            scheduler.step(val_loss)

            # Save training history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_cnn_lstm_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Load best model and evaluate
        self.model.load_state_dict(torch.load('best_cnn_lstm_model.pth'))
        self.is_trained = True

        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            # Test accuracy
            test_outputs = self.model(X_test)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_accuracy = (test_predicted == y_test).float().mean().item()

        training_results = {
            'success': True,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'best_val_loss': best_val_loss,
            'total_epochs': len(self.training_history),
            'feature_count': input_size,
            'sequence_count': len(X),
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }

        logger.info(f"CNN-LSTM training completed - Test Acc: {test_accuracy:.3f}")
        return training_results

    def generate_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signals using trained CNN-LSTM model"""
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained. Training now...")
            self.train_model(data)

        logger.info("Generating CNN-LSTM trading signals...")

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

                    # Enhanced signal generation
                    min_confidence = 0.4
                    strong_confidence = 0.65

                    if predicted_class == 1 and max_prob > min_confidence:  # Buy
                        signals.iloc[i] = 1
                        strength = prob_buy - max(prob_sell, prob_hold)
                        if max_prob > strong_confidence:
                            signal_strength.iloc[i] = strength * 1.2
                        else:
                            signal_strength.iloc[i] = strength

                    elif predicted_class == 2 and max_prob > min_confidence:  # Sell
                        signals.iloc[i] = -1
                        strength = prob_sell - max(prob_buy, prob_hold)
                        if max_prob > strong_confidence:
                            signal_strength.iloc[i] = -strength * 1.2
                        else:
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
            'strategy_name': 'CNN-LSTM',
            'model_type': 'Deep Learning'
        }

        logger.info(f"CNN-LSTM signals generated - Hit Rate: {hit_rate:.1f}%, "
                   f"Buy: {results['signal_distribution']['buy']}, "
                   f"Sell: {results['signal_distribution']['sell']}")

        return results