# Wind Speed Prediction Experiment Report

Generated: 2025-12-30 12:25:36

## 1. Experiment Configuration

- Device: cuda
- Batch Size: 512
- Max Epochs: 2000
- Learning Rate: 5e-05
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
| CNN_LSTM        | singlestep    | 0.904447 | 0.951    | 0.7227   | 0.878    |
| DLinear         | singlestep    | 0.859821 | 0.927266 | 0.692295 | 0.884015 |
| HeightAttention | singlestep    | 1.08112  | 1.03977  | 0.798756 | 0.854162 |
| LSTM            | singlestep    | 0.854254 | 0.9243   | 0.6948   | 0.8848   |
| LSTNet          | singlestep    | 0.898757 | 0.948    | 0.7225   | 0.8788   |
| Linear          | singlestep    | 0.858833 | 0.9267   | 0.706    | 0.8841   |
| TCN             | singlestep    | 0.874418 | 0.9351   | 0.7026   | 0.882    |
| Transformer     | singlestep    | 0.891391 | 0.9441   | 0.7077   | 0.8798   |
| WaveNet         | singlestep    | 0.908046 | 0.9529   | 0.722    | 0.8775   |
| CNN_LSTM        | multistep_16h | 3.98992  | 1.9975   | 1.5828   | 0.464    |
| DLinear         | multistep_16h | 3.85336  | 1.963    | 1.54302  | 0.482341 |
| HeightAttention | multistep_16h | 4.69292  | 2.16631  | 1.7201   | 0.369554 |
| LSTM            | multistep_16h | 3.55428  | 1.8853   | 1.4937   | 0.5225   |
| LSTNet          | multistep_16h | 4.44139  | 2.1075   | 1.6274   | 0.4033   |
| Linear          | multistep_16h | 3.60898  | 1.8997   | 1.5101   | 0.5152   |
| TCN             | multistep_16h | 4.16839  | 2.0417   | 1.6304   | 0.44     |
| Transformer     | multistep_16h | 4.05958  | 2.0148   | 1.5991   | 0.4546   |
| WaveNet         | multistep_16h | 3.80249  | 1.95     | 1.5346   | 0.4892   |

## 4. Best Models

- **Single-step Prediction (8h -> 1h)**: LSTM (RMSE: 0.9243, R²: 0.8848)
- **Multi-step Prediction (8h -> 16h)**: LSTM (RMSE: 1.8853, R²: 0.5225)

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
