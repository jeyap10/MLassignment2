# Pulmanory Heart Disease Prediction using ML

---

## a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict the presence or absence of heart disease based on clinical features. The project also demonstrates deployment of these models in an interactive Streamlit web application.

---

## b. Dataset Description

- **Dataset Name:** Heart Disease Dataset (UCI / Kaggle)
- **Number of Instances:** 900+
- **Number of Features:** 13
- **Target Variable:** `target` (0 = No heart disease, 1 = Heart disease)
- **Type:** Binary Classification
- **Description:** The dataset contains clinical attributes such as age, sex, blood pressure, cholesterol, maximum heart rate, and others, which are used to predict the presence of heart disease.

---

## c. Models Used and Evaluation Metrics

|                          |   Accuracy |   AUC |   Precision |   Recall |   F1 |   MCC |
|:-------------------------|-----------:|------:|------------:|---------:|-----:|------:|
| Logistic Regression      |       0.87 |  0.92 |        0.88 |     0.85 | 0.86 |  0.74 |
| Decision Tree            |       0.83 |  0.85 |        0.81 |     0.8  | 0.8  |  0.65 |
| KNN                      |       0.85 |  0.87 |        0.84 |     0.83 | 0.83 |  0.69 |
| Naive Bayes              |       0.82 |  0.86 |        0.82 |     0.8  | 0.81 |  0.63 |
| Random Forest (Ensemble) |       0.89 |  0.94 |        0.9  |     0.87 | 0.88 |  0.78 |
| XGBoost (Ensemble)       |       0.9  |  0.95 |        0.91 |     0.88 | 0.89 |  0.8  |

---

##  Observations on Model Performance

- **ML Model Name             **: |Observation about model performance
- **-------------------------------------------------------------------------------------------------------------------------------------**: 
- **Logistic Regression       **: |Performs well on linearly separable features. Fast training but may underperform on non-linear relationships.
- **Decision Tree             **: |Easy to interpret, can overfit on training data if not tuned. Moderate accuracy on test set.
- **KNN                       **: |Sensitive to feature scaling. Performance depends on choice of K and distance metric.
- **Naive Bayes               **: |Assumes feature independence, computationally efficient but may be less accurate if features are correlated.
- **Random Forest (Ensemble)  **: |Handles non-linearity and reduces overfitting. Performs very well on most metrics.
- **XGBoost (Ensemble)        **: |Best overall performance due to boosting. Strong generalization and high AUC.

---

## How to Run

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Upload a test CSV file and select a model from the dropdown to see predictions and evaluation metrics.
