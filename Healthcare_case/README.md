# Demo for the HH use cases

The Pilot's remo: [Stress_Detection](https://github.com/StamatisOrfanos/Stress_Detection?tab=readme-ov-file)

## Datasets

- [Nurse Stress Prediction Wearable Sensors](https://www.kaggle.com/datasets/priyankraval/nurse-stress-prediction-wearable-sensors): Similarly in this dataset we use the heart rate data in order to calculate heart rate variability that is going to be used for the models.

- [SWELL dataset](https://www.kaggle.com/datasets/qiriro/swell-heart-rate-variability-hrv): This dataset comprises of heart rate variability (HRV) indices computed from the multimodal SWELL knowledge work (SWELL-KW) dataset for research on stress and user modeling, see SWELL-KW.

- [Stress-Predict-Dataset](https://github.com/italha-d/Stress-Predict-Dataset): This dataset is associated with "Stress Monitoring Using Wearable Sensors: A Pilot Study and Stress-Predict Dataset" paper.

- [Heart Rate Prediction to Monitor Stress Level](https://www.kaggle.com/datasets/vinayakshanawad/heart-rate-prediction-to-monitor-stress-level): The data comprises various attributes taken from signals measured using ECG recorded for different individuals having different heart rates at the time the measurement was taken. These various features contribute to the heart rate at the given instant of time for the individual.

## DL Model

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 2)              │             6 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 4)              │            12 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 8)              │            40 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 3)              │            27 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 85 (340.00 B)
 Trainable params: 85 (340.00 B)
 Non-trainable params: 0 (0.00 B)

 accuracy: 0.7377 - loss: 0.7406