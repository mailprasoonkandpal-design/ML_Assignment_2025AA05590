
# ML_Assignment2
=======
# ML Assignment 2

## Problem Statement
The objective of this project is to build and compare multiple machine learning classification models to predict whether a customer will respond positively to a marketing campaign (target class). The goal is to identify the model that provides the best predictive performance based on evaluation metrics such as Accuracy, AUC, Precision, Recall, F1-Score, and MCC. The task also involves analyzing model behavior, strengths, weaknesses, and suitability for real-world decision-making.

## Dataset Description
Contains 45,211 customer records with 17 columns.
Target variable predicts whether a client subscribed to a term deposit (Yes/No).
Includes both numerical (age, balance, duration, etc.) and categorical (job, marital, education, etc.) features.
Dataset is imbalanced majority class is No (~88%).
Used for classification modeling in banking marketing analysis.

## Models Used
Logistic Regression: Linear classifier that outputs probabilities; effective for simple, linearly separable data.
Decision Tree: Rule-based model that is easy to interpret but can overfit.
KNN: Predicts based on nearest neighbors; sensitive to scaling and distance metrics.
Naive Bayes: Fast probabilistic model assuming feature independence.
Random Forest: Ensemble of trees that improves accuracy and reduces overfitting.
XGBoost: Advanced boosting algorithm with strong performance and generalization.

## Model Comparison Table
| Model Name           | Accuracy | AUC      | Precision | Recall   | F1 Score | MCC     |
|----------------------|----------|----------|-----------|----------|----------|---------|
| Logistic Regression  | 0.8914   | 0.8726   | 0.5945    | 0.2259   | 0.3274   | 0.3205  |
| Decision Tree        | 0.8769   | 0.7044   | 0.4743    | 0.4792   | 0.4767   | 0.4070  |
| KNN                  | 0.8923   | 0.8089   | 0.5717    | 0.3166   | 0.4754   | 0.3724  |
| Naive Bayes          | 0.8445   | 0.8160   | 0.3659    | 0.4490   | 0.4323   | 0.3171  |
| Random Forest        | 0.9063   | 0.9246   | 0.6572    | 0.4168   | 0.5101   | 0.4758  |
| XGBoost              | 0.9058   | 0.9267   | 0.6281    | 0.4773   | 0.5424   | 0.4968  |


## Observations
| ML Model Name | Observation |
|---------------|-------------|
| Logistic Regression | Achieved good overall accuracy and AUC, but very low recall indicates it struggles to correctly identify positive cases. Suitable when false positives must be minimized, but not ideal for detecting rare positives. |
| Decision Tree | Balanced precision and recall compared to Logistic Regression, but lower AUC suggests weaker generalization. Performs reasonably but is prone to overfitting and instability. |
| KNN | High accuracy but moderate recall and MCC. Indicates good overall predictions but weaker performance in identifying minority class. Performance depends heavily on distance metric and scaling. |
| Naive Bayes | Lowest accuracy among models but relatively balanced recall. Shows stable but simple performance; independence assumption limits predictive power. |
| Random Forest (Ensemble) | Highest accuracy with strong AUC and MCC, indicating excellent overall predictive performance and class discrimination. Ensemble averaging reduces overfitting and improves robustness. |
| XGBoost (Ensemble) | Best MCC and highest AUC among all models, showing superior classification quality and best balance between precision and recall. Demonstrates strongest overall performance. |

