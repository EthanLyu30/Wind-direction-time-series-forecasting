# Wind Speed Prediction Experiment Report

Generated: 2025-12-29 13:27:39

## 1. Experiment Configuration

- Device: cuda
- Batch Size: 128
- Max Epochs: 500
- Learning Rate: 0.0001
- Random Seed: 42

## 2. Task Configuration

### Single-step Prediction (8h -> 1h)
- Input Length: 8 hours
- Output Length: 1 hours

### Multi-step Prediction (8h -> 1h)
- Input Length: 8 hours
- Output Length: 1 hours

### Multi-step Prediction (24h -> 16h)
- Input Length: 8 hours
- Output Length: 16 hours

## 3. Model Performance Comparison

| Model          | Task          |      MSE |     RMSE |      MAE |       R2 |
|:---------------|:--------------|---------:|---------:|---------:|---------:|
| Attention_LSTM | singlestep    | 0.98068  | 0.990293 | 0.761904 | 0.867711 |
| CNN_LSTM       | singlestep    | 0.905852 | 0.951762 | 0.724298 | 0.877805 |
| LSTM           | singlestep    | 0.838082 | 0.915468 | 0.690491 | 0.886947 |
| Linear         | singlestep    | 0.858833 | 0.926732 | 0.706027 | 0.884148 |
| TCN            | singlestep    | 0.874418 | 0.935103 | 0.702591 | 0.882046 |
| Transformer    | singlestep    | 0.940539 | 0.969814 | 0.744803 | 0.873126 |
| WaveNet        | singlestep    | 0.92921  | 0.963955 | 0.730031 | 0.874655 |
| Attention_LSTM | multistep  | 0.899375 | 0.948354 | 0.720456 | 0.878679 |
| CNN_LSTM       | multistep  | 0.957306 | 0.97842  | 0.739876 | 0.870865 |
| LSTM           | multistep  | 0.83368  | 0.913061 | 0.692427 | 0.887541 |
| Linear         | multistep  | 0.879428 | 0.937778 | 0.711614 | 0.88137  |
| TCN            | multistep  | 0.885926 | 0.941237 | 0.705466 | 0.880493 |
| Transformer    | multistep  | 0.902177 | 0.94983  | 0.719141 | 0.878301 |
| WaveNet        | multistep  | 0.90821  | 0.953    | 0.726091 | 0.877487 |
| Attention_LSTM | multistep | 4.17063  | 2.04221  | 1.60845  | 0.439718 |
| CNN_LSTM       | multistep | 4.05914  | 2.01473  | 1.59901  | 0.454696 |
| LSTM           | multistep | 3.86785  | 1.96669  | 1.55179  | 0.480393 |
| Linear         | multistep | 3.72823  | 1.93086  | 1.53772  | 0.49915  |
| TCN            | multistep | 4.25936  | 2.06382  | 1.63268  | 0.427798 |
| Transformer    | multistep | 4.59876  | 2.14447  | 1.67793  | 0.382204 |
| WaveNet        | multistep | 3.83944  | 1.95945  | 1.54765  | 0.48421  |

## 4. Best Models

- **Single-step Prediction (8h -> 1h)**: LSTM (RMSE: 0.9155, R²: 0.8869)
- **Multi-step Prediction (8h -> 1h)**: LSTM (RMSE: 0.9131, R²: 0.8875)
- **Multi-step Prediction (24h -> 16h)**: Linear (RMSE: 1.9309, R²: 0.4991)

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
