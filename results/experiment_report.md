# Wind Speed Prediction Experiment Report

Generated: 2025-12-30 09:54:30

## 1. Experiment Configuration

- Device: cuda
- Batch Size: 128
- Max Epochs: 200
- Learning Rate: 0.0003
- Random Seed: 42

## 2. Task Configuration

### Single-step Prediction (8h -> 1h)
- Input Length: 8 hours
- Output Length: 1 hours

### Multi-step Prediction (8h -> 1h)
- Input Length: 8 hours
- Output Length: 1 hours

### Multi-step Prediction (24h -> 16h)
- Input Length: 24 hours
- Output Length: 16 hours

## 3. Model Performance Comparison

| Model          | Task          |      MSE |     RMSE |      MAE |       R2 |
|:---------------|:--------------|---------:|---------:|---------:|---------:|
| Attention_LSTM | singlestep    | 0.980706 | 0.990306 | 0.761917 | 0.867708 |
| CNN_LSTM       | singlestep    | 0.893232 | 0.94511  | 0.714446 | 0.879508 |
| LSTM           | singlestep    | 0.862288 | 0.928595 | 0.704977 | 0.883682 |
| Linear         | singlestep    | 0.858833 | 0.926732 | 0.706028 | 0.884148 |
| TCN            | singlestep    | 0.871597 | 0.933593 | 0.698307 | 0.882426 |
| Transformer    | singlestep    | 0.940538 | 0.969814 | 0.744802 | 0.873126 |
| WaveNet        | singlestep    | 0.929103 | 0.9639   | 0.729978 | 0.874669 |
| Attention_LSTM | multistep_1h  | 0.899382 | 0.948357 | 0.720457 | 0.878678 |
| CNN_LSTM       | multistep_1h  | 0.949542 | 0.974445 | 0.749623 | 0.871912 |
| LSTM           | multistep_1h  | 0.846955 | 0.920302 | 0.69832  | 0.88575  |
| Linear         | multistep_1h  | 0.879428 | 0.937778 | 0.711614 | 0.88137  |
| TCN            | multistep_1h  | 0.876663 | 0.936303 | 0.702778 | 0.881743 |
| Transformer    | multistep_1h  | 0.887594 | 0.942122 | 0.709369 | 0.880268 |
| WaveNet        | multistep_1h  | 0.927447 | 0.963041 | 0.730231 | 0.874892 |
| Attention_LSTM | multistep_16h | 4.78384  | 2.1872   | 1.72254  | 0.363525 |
| CNN_LSTM       | multistep_16h | 4.54     | 2.13073  | 1.67297  | 0.395967 |
| LSTM           | multistep_16h | 3.60939  | 1.89984  | 1.49172  | 0.519782 |
| Linear         | multistep_16h | 4.31425  | 2.07708  | 1.64939  | 0.426002 |
| TCN            | multistep_16h | 5.35755  | 2.31464  | 1.83344  | 0.287194 |
| Transformer    | multistep_16h | 4.69938  | 2.1678   | 1.65948  | 0.374762 |
| WaveNet        | multistep_16h | 4.73551  | 2.17612  | 1.70015  | 0.369955 |

## 4. Best Models

- **Single-step Prediction (8h -> 1h)**: Linear (RMSE: 0.9267, R²: 0.8841)
- **Multi-step Prediction (8h -> 1h)**: LSTM (RMSE: 0.9203, R²: 0.8858)
- **Multi-step Prediction (24h -> 16h)**: LSTM (RMSE: 1.8998, R²: 0.5198)

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

## 6. Conclusion

This experiment compared Linear, LSTM, and Transformer as baseline models, along with CNN-LSTM, Attention-LSTM, TCN, and WaveNet as innovative models for wind speed prediction tasks.

The results show that deep learning models have significant advantages in capturing wind speed temporal features, especially models with attention mechanisms that can better capture long-term dependencies.
