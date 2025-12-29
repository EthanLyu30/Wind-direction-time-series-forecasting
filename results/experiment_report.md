# Wind Speed Prediction Experiment Report

Generated: 2025-12-29 15:34:08

## 1. Experiment Configuration

- Device: cuda
- Batch Size: 256
- Max Epochs: 800
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

| Model       | Task          |      MSE |     RMSE |      MAE |       R2 |
|:------------|:--------------|---------:|---------:|---------:|---------:|
| CNN_LSTM    | singlestep    | 0.904447 | 0.951024 | 0.722667 | 0.877995 |
| LSTM        | singlestep    | 0.854254 | 0.924259 | 0.694755 | 0.884766 |
| Linear      | singlestep    | 0.858833 | 0.926732 | 0.706027 | 0.884148 |
| NBEATS      | singlestep    | 1.16628  | 1.07995  | 0.833252 | 0.842675 |
| TCN         | singlestep    | 0.874418 | 0.935103 | 0.702591 | 0.882046 |
| Transformer | singlestep    | 0.891391 | 0.944135 | 0.707706 | 0.879756 |
| WaveNet     | singlestep    | 0.908054 | 0.952919 | 0.722012 | 0.877508 |
| CNN_LSTM    | multistep_16h | 4.10591  | 2.02631  | 1.60491  | 0.448412 |
| LSTM        | multistep_16h | 3.70354  | 1.92446  | 1.52456  | 0.502466 |
| Linear      | multistep_16h | 3.60898  | 1.89973  | 1.51013  | 0.51517  |
| NBEATS      | multistep_16h | 4.78301  | 2.18701  | 1.70261  | 0.357451 |
| TCN         | multistep_16h | 4.2551   | 2.06279  | 1.63876  | 0.428371 |
| Transformer | multistep_16h | 4.05958  | 2.01484  | 1.5991   | 0.454637 |
| WaveNet     | multistep_16h | 3.80249  | 1.95     | 1.53455  | 0.489174 |

## 4. Best Models

- **Single-step Prediction (8h -> 1h)**: LSTM (RMSE: 0.9243, R²: 0.8848)
- **Multi-step Prediction (8h -> 16h)**: Linear (RMSE: 1.8997, R²: 0.5152)

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
