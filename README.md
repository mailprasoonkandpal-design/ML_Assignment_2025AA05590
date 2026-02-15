<<<<<<< HEAD
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
Model Name	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.891408	0.872583	0.594527	0.225898	0.327397	0.320453
Decision Tree	0.876921	0.704412	0.474275	0.479206	0.476728	0.406996
KNN	0.892292	0.808924	0.571672	0.316635	0.47543	0.37237
Naive Bayes	0.844521	0.815968	0.365948	0.44896	0.43226	0.317083
Random Forest	0.906336	0.924555	0.657228	0.416824	0.510121	0.475838
XGBoost	0.905783	0.926749	0.628109	0.477316	0.542427	0.496752

## Observations
ML Model Name	Observation about model performance
Logistic Regression	Achieved good overall accuracy and AUC, but very low recall indicates it struggles to correctly identify positive cases. Suitable when false positives must be minimised, but not ideal for detecting rare positives.
Decision Tree	Balanced precision and recall compared to Logistic Regression, but lower AUC suggests weaker generalisation. Performs reasonably but is prone to overfitting and instability.
KNN	High accuracy but moderate recall and MCC. Indicates good overall predictions but weaker performance in identifying minority class. Performance depends heavily on distance metric and scaling.
Naive Bayes	Lowest accuracy among models but relatively balanced recall. Shows stable but simple performance; independence assumption limits predictive power.
Random Forest (Ensemble)	Highest accuracy with strong AUC and MCC, indicating excellent overall predictive performance and class discrimination. Ensemble averaging reduces overfitting and improves robustness.
XGBoost (Ensemble)	Best MCC and highest AUC among all models, showing superior classification quality and best balance between precision and recall. Demonstrates strongest overall performance.
