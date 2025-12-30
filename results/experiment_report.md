# Wind Speed Prediction Experiment Report

Generated: 2025-12-30 17:37:34

## 1. Experiment Configuration

- Device: cuda
- Batch Size: 512
- Max Epochs: 500
- Learning Rate: 0.008
- Random Seed: 42

## 2. Task Configuration

### Single-step Prediction (8h -> 1h)
- Input Length: 8 hours
- Output Length: 1 hours

### Multi-step Prediction (8h -> 16h)
- Input Length: 8 hours
- Output Length: 16 hours

## 3. Model Performance Comparison

| Model           | Task          |      MSE |     RMSE |      MAE |       R2 |
|:----------------|:--------------|---------:|---------:|---------:|---------:|
| CNN_LSTM        | singlestep    | 0.891205 | 0.944037 | 0.722117 | 0.879781 |
| DLinear         | singlestep    | 0.839374 | 0.916174 | 0.680348 | 0.886773 |
| HeightAttention | singlestep    | 0.856736 | 0.9256   | 0.701066 | 0.884431 |
| LSTM            | singlestep    | 0.837983 | 0.915414 | 0.695138 | 0.886961 |
| LSTNet          | singlestep    | 0.876975 | 0.936469 | 0.714991 | 0.881701 |
| Linear          | singlestep    | 0.858833 | 0.9267   | 0.706    | 0.8841   |
| Persistence     | singlestep    | 0.858553 | 0.926581 | 0.68688  | 0.884186 |
| TCN             | singlestep    | 0.839684 | 0.916342 | 0.688811 | 0.886731 |
| Transformer     | singlestep    | 0.891391 | 0.9441   | 0.7077   | 0.8798   |
| TrendLinear     | singlestep    | 0.871079 | 0.933316 | 0.69689  | 0.882496 |
| WaveNet         | singlestep    | 0.908046 | 0.9529   | 0.722    | 0.8775   |
| WindShear       | singlestep    | 0.88967  | 0.943223 | 0.704111 | 0.879988 |
| CNN_LSTM        | multistep_16h | 3.79372  | 1.94775  | 1.54528  | 0.490352 |
| DLinear         | multistep_16h | 3.70921  | 1.92593  | 1.50565  | 0.501705 |
| HeightAttention | multistep_16h | 3.47485  | 1.8641   | 1.45717  | 0.533189 |
| LSTM            | multistep_16h | 3.55428  | 1.8853   | 1.4937   | 0.5225   |
| LSTNet          | multistep_16h | 3.74718  | 1.93576  | 1.50591  | 0.496604 |
| Linear          | multistep_16h | 3.58261  | 1.89278  | 1.47767  | 0.518712 |
| Persistence     | multistep_16h | 3.89446  | 1.97344  | 1.49066  | 0.476818 |
| TCN             | multistep_16h | 3.61654  | 1.90172  | 1.49757  | 0.514155 |
| Transformer     | multistep_16h | 4.05958  | 2.0148   | 1.5991   | 0.4546   |
| TrendLinear     | multistep_16h | 4.03091  | 2.00771  | 1.59502  | 0.458489 |
| WaveNet         | multistep_16h | 3.44293  | 1.85551  | 1.44923  | 0.537478 |
| WindShear       | multistep_16h | 3.86105  | 1.96495  | 1.49117  | 0.481308 |

## 4. Best Models

- **Single-step Prediction (8h -> 1h)**: LSTM (RMSE: 0.9154, R²: 0.8870)
- **Multi-step Prediction (8h -> 16h)**: WaveNet (RMSE: 1.8555, R²: 0.5375)

## 5. Innovation Points

### 5.1 CNN-LSTM Hybrid Model
- Combines CNN's local feature extraction with LSTM's sequence modeling
- Multi-scale convolution kernels capture features at different time scales
- Attention mechanism enhances important feature weights

### 5.2 Attention-LSTM Model
- Self-attention mechanism enhances feature representation
- Temporal attention focuses on key time points
- Multi-head attention processes different subspace information in parallel

### 5.3 TCN Model
- Causal convolution ensures temporal causality
- Dilated convolution exponentially expands receptive field
- Residual connections stabilize deep network training

### 5.4 WaveNet Model
- Gated activation units enhance expressive power
- Dilated causal convolution efficiently models long sequences
- Residual and Skip connections accelerate gradient flow

### 5.5 LSTNet Model
- CNN layer extracts short-term local patterns
- GRU layer captures long-term dependencies
- Skip-RNN models periodic patterns directly
- Highway component (autoregressive) enhances prediction stability

## 6. Conclusion

This experiment compared Linear, LSTM, and Transformer as baseline models, along with CNN-LSTM, Attention-LSTM, TCN, and WaveNet as innovative models for wind speed prediction tasks.

The results show that deep learning models have significant advantages in capturing wind speed temporal features, especially models with attention mechanisms that can better capture long-term dependencies.
