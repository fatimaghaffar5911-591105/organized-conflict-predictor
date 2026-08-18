# 🌍 Conflict Risk Prediction

A machine learning application exploring whether previous-year socioeconomic conditions can help predict organized state-based conflict.

## Model

The application uses a Random Forest classifier with six lagged socioeconomic indicators:

- GDP per capita
- GDP growth
- Inflation
- Population
- Unemployment
- Military expenditure

## Results

Lagged Random Forest:

- Accuracy: 92.49%
- Conflict precision: 86%
- Conflict recall: 76%
- Conflict F1-score: 81%

Country-held-out evaluation:

- Accuracy: 83.12%
- Conflict recall: 21%
- Conflict F1-score: 31%

The stricter country-held-out evaluation demonstrates the importance of testing whether a model generalizes to countries not seen during training.

## Disclaimer

This is an academic machine-learning project. Predictions should not be interpreted as causal explanations or definitive forecasts.

## Author

BS International Relations student exploring the intersection of International Relations, Geopolitics, Political Analysis, Data Science and Machine Learning.
