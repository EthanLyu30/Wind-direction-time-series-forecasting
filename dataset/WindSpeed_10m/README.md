---
dataset_info:
  features:
  - name: Date & Time Stamp
    dtype: string
  - name: SpeedAvg
    dtype: float64
  - name: SpeedMax
    dtype: float64
  - name: DirectionAvg
    dtype: int64
  - name: TemperatureAvg
    dtype: float64
  - name: TemperatureMax
    dtype: float64
  - name: PressureAvg
    dtype: float64
  - name: PressureMax
    dtype: float64
  - name: HumidtyAvg
    dtype: float64
  - name: HumityMax
    dtype: float64
  - name: height
    dtype: int64
  splits:
  - name: train
    num_bytes: 837334
    num_examples: 8458
  - name: val
    num_bytes: 104223
    num_examples: 1057
  - name: test
    num_bytes: 104307
    num_examples: 1058
  download_size: 238387
  dataset_size: 1045864
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: val
    path: data/val-*
  - split: test
    path: data/test-*
---
