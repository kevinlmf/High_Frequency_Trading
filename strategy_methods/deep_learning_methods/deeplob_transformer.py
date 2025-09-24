"""
DeepLOB + Transformer Strategy for HFT
结合DeepLOB的CNN架构和Transformer的注意力机制来预测价格走势
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DeepLOBCNN(nn.Module):
    """
    简化的DeepLOB CNN模块 - 提取限价订单簿特征
    """

    def __init__(self, input_dim: int = 40, output_dim: int = 64):
        super(DeepLOBCNN, self).__init__()

        self.output_dim = output_dim

        # 使用1D卷积来处理时序数据
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, output_dim, kernel_size=3, padding=1)

        # 批归一化
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(output_dim)

        # Dropout和激活函数
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        前向传播
        Args:
            x: (batch_size, seq_len, features)
        Returns:
            features: (batch_size, seq_len, output_dim)
        """
        # 转置用于1D卷积: (batch_size, features, seq_len)
        x = x.transpose(1, 2)

        # CNN特征提取
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        x = self.relu(self.bn3(self.conv3(x)))

        # 转回原始格式: (batch_size, seq_len, output_dim)
        x = x.transpose(1, 2)

        return x


class TransformerPredictor(nn.Module):
    """
    Transformer预测模块 - 基于时序注意力预测价格走势
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super(TransformerPredictor, self).__init__()

        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(5000, d_model))

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 3)  # 3个输出：买/卖/持有概率
        )

    def forward(self, x, mask=None):
        """
        前向传播
        Args:
            x: (batch_size, seq_len, input_dim)
            mask: 注意力掩码
        Returns:
            predictions: (batch_size, seq_len, 3)
        """
        seq_len = x.size(1)

        # 输入投影
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)

        # 添加位置编码
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)

        # Transformer编码
        x = self.transformer(x, src_key_padding_mask=mask)

        # 预测输出
        predictions = self.output_projection(x)

        return predictions


class DeepLOBTransformer(nn.Module):
    """
    DeepLOB + Transformer 组合模型
    """

    def __init__(
        self,
        lob_input_dim: int = 40,
        market_feature_dim: int = 20,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super(DeepLOBTransformer, self).__init__()

        # DeepLOB CNN特征提取器
        cnn_output_dim = d_model // 2
        self.deeplob_cnn = DeepLOBCNN(lob_input_dim, cnn_output_dim)

        # 市场特征编码器
        self.market_encoder = nn.Linear(market_feature_dim, d_model // 2)

        # 特征融合层
        total_dim = cnn_output_dim + d_model // 2
        self.feature_fusion = nn.Linear(total_dim, d_model)

        # Transformer预测器
        self.transformer = TransformerPredictor(
            input_dim=d_model,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, lob_data, market_features, mask=None):
        """
        前向传播
        Args:
            lob_data: (batch_size, seq_len, lob_input_dim) 限价订单簿数据
            market_features: (batch_size, seq_len, market_feature_dim) 市场特征
            mask: 注意力掩码
        Returns:
            predictions: (batch_size, seq_len, 3) 预测概率
        """
        # LOB特征提取
        lob_features = self.deeplob_cnn(lob_data)

        # 市场特征编码
        market_encoded = F.relu(self.market_encoder(market_features))

        # 特征融合
        combined_features = torch.cat([lob_features, market_encoded], dim=-1)
        fused_features = F.relu(self.feature_fusion(combined_features))

        # Transformer预测
        predictions = self.transformer(fused_features, mask)

        return predictions


class DeepLOBTransformerStrategy:
    """
    DeepLOB + Transformer 策略实现
    """

    def __init__(
        self,
        lob_input_dim: int = 40,
        market_feature_dim: int = 20,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        sequence_length: int = 100,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length

        # 初始化模型
        self.model = DeepLOBTransformer(
            lob_input_dim=lob_input_dim,
            market_feature_dim=market_feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(self.device)

        # 优化器和损失函数
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        self.criterion = nn.CrossEntropyLoss()

        # 训练历史
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        logger.info(f"DeepLOB + Transformer Strategy initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def prepare_data(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        准备训练数据
        """
        # 创建LOB数据（简化版，实际中需要真实的订单簿数据）
        lob_features = ['open', 'high', 'low', 'close', 'volume']
        lob_data = data[lob_features].values

        # 标准化
        lob_data = (lob_data - lob_data.mean(axis=0)) / (lob_data.std(axis=0) + 1e-8)

        # 扩展到40维（模拟订单簿的10档买卖盘）
        lob_extended = np.tile(lob_data, (1, 8))  # 5*8 = 40

        # 市场特征
        market_features = features.values
        market_features = (market_features - market_features.mean(axis=0)) / (market_features.std(axis=0) + 1e-8)

        # 创建序列
        sequences_lob = []
        sequences_market = []
        labels = []

        for i in range(self.sequence_length, len(lob_extended)):
            sequences_lob.append(lob_extended[i-self.sequence_length:i])
            sequences_market.append(market_features[i-self.sequence_length:i])

            # 简单标签：基于下一期价格变化
            next_price = data['close'].iloc[i]
            current_price = data['close'].iloc[i-1]
            price_change = (next_price - current_price) / current_price

            if price_change > 0.001:  # 上涨
                label = 0
            elif price_change < -0.001:  # 下跌
                label = 1
            else:  # 持有
                label = 2

            labels.append(label)

        return {
            'lob_data': torch.FloatTensor(sequences_lob).to(self.device),
            'market_features': torch.FloatTensor(sequences_market).to(self.device),
            'labels': torch.LongTensor(labels).to(self.device)
        }

    def train(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        训练模型
        """
        logger.info(f"开始训练 DeepLOB + Transformer 模型...")

        # 准备数据
        dataset = self.prepare_data(data, features)

        # 划分训练和验证集
        val_size = int(len(dataset['lob_data']) * validation_split)
        train_size = len(dataset['lob_data']) - val_size

        train_data = {
            'lob_data': dataset['lob_data'][:train_size],
            'market_features': dataset['market_features'][:train_size],
            'labels': dataset['labels'][:train_size]
        }

        val_data = {
            'lob_data': dataset['lob_data'][train_size:],
            'market_features': dataset['market_features'][train_size:],
            'labels': dataset['labels'][train_size:]
        }

        # 训练循环
        self.model.train()
        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(train_data, batch_size)
            val_loss, val_acc = self._validate_epoch(val_data, batch_size)

            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)

            # 学习率调度
            self.scheduler.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        logger.info("训练完成！")
        return self.training_history

    def _train_epoch(self, train_data: Dict[str, torch.Tensor], batch_size: int) -> Tuple[float, float]:
        """训练一个epoch"""
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for i in range(0, len(train_data['lob_data']), batch_size):
            batch_lob = train_data['lob_data'][i:i+batch_size]
            batch_market = train_data['market_features'][i:i+batch_size]
            batch_labels = train_data['labels'][i:i+batch_size]

            self.optimizer.zero_grad()

            # 前向传播
            predictions = self.model(batch_lob, batch_market)

            # 只使用最后一个时间步的预测
            predictions = predictions[:, -1, :]
            loss = self.criterion(predictions, batch_labels)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(predictions.data, 1)
            total_correct += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

        return total_loss / (total_samples / batch_size), total_correct / total_samples

    def _validate_epoch(self, val_data: Dict[str, torch.Tensor], batch_size: int) -> Tuple[float, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for i in range(0, len(val_data['lob_data']), batch_size):
                batch_lob = val_data['lob_data'][i:i+batch_size]
                batch_market = val_data['market_features'][i:i+batch_size]
                batch_labels = val_data['labels'][i:i+batch_size]

                predictions = self.model(batch_lob, batch_market)
                predictions = predictions[:, -1, :]
                loss = self.criterion(predictions, batch_labels)

                total_loss += loss.item()
                _, predicted = torch.max(predictions.data, 1)
                total_correct += (predicted == batch_labels).sum().item()
                total_samples += batch_labels.size(0)

        self.model.train()
        return total_loss / (total_samples / batch_size), total_correct / total_samples

    def predict(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        生成交易信号
        """
        self.model.eval()

        # 准备数据
        dataset = self.prepare_data(data, features)

        signals = []
        with torch.no_grad():
            for i in range(len(dataset['lob_data'])):
                lob_input = dataset['lob_data'][i:i+1]
                market_input = dataset['market_features'][i:i+1]

                prediction = self.model(lob_input, market_input)
                prediction = prediction[:, -1, :]  # 最后时间步

                # 转换为信号
                probabilities = F.softmax(prediction, dim=1)
                signal_class = torch.argmax(probabilities, dim=1).item()

                if signal_class == 0:  # 买入
                    signal = 1
                elif signal_class == 1:  # 卖出
                    signal = -1
                else:  # 持有
                    signal = 0

                signals.append(signal)

        return pd.Series(signals, index=data.index[-len(signals):])

    def save_model(self, filepath: str) -> None:
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, filepath)
        logger.info(f"模型已保存到 {filepath}")

    def load_model(self, filepath: str) -> None:
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', self.training_history)
        logger.info(f"模型已从 {filepath} 加载")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取模型性能指标"""
        if not self.training_history['train_loss']:
            return {}

        return {
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_val_loss': self.training_history['val_loss'][-1],
            'final_train_acc': self.training_history['train_acc'][-1],
            'final_val_acc': self.training_history['val_acc'][-1],
            'best_val_acc': max(self.training_history['val_acc']),
            'total_epochs': len(self.training_history['train_loss'])
        }


if __name__ == "__main__":
    # 测试代码
    from pathlib import Path
    import sys

    # 添加路径
    sys.path.append(str(Path(__file__).parent.parent.parent))

    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000

    # 模拟价格数据
    test_data = pd.DataFrame({
        'open': np.random.randn(n_samples).cumsum() + 100,
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n_samples)
    })

    # 模拟技术指标
    test_features = pd.DataFrame({
        'rsi': np.random.uniform(0, 100, n_samples),
        'macd': np.random.randn(n_samples),
        'bb_upper': np.random.randn(n_samples) + 105,
        'bb_lower': np.random.randn(n_samples) + 95,
        'volume_sma': np.random.randint(5000, 15000, n_samples)
    })

    # 扩展到20维
    for i in range(15):
        test_features[f'feature_{i}'] = np.random.randn(n_samples)

    print("创建 DeepLOB + Transformer 策略...")
    strategy = DeepLOBTransformerStrategy(
        lob_input_dim=40,
        market_feature_dim=20,
        d_model=128,  # 减小模型以快速测试
        nhead=4,
        num_layers=3,
        sequence_length=50
    )

    print("开始训练...")
    training_results = strategy.train(
        test_data,
        test_features,
        epochs=20,
        batch_size=16
    )

    print("生成交易信号...")
    signals = strategy.predict(test_data, test_features)
    print(f"生成了 {len(signals)} 个信号")
    print(f"信号分布: {signals.value_counts().to_dict()}")

    print("性能指标:")
    metrics = strategy.get_performance_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")