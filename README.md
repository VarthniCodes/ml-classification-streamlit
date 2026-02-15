# ml-classification-streamlit
# Machine Learning Classification Models with Streamlit Deployment

## a. Problem Statement
The objective of this project is to implement multiple machine learning classification models on a public dataset, evaluate their performance using various metrics, and deploy an interactive web application using Streamlit for model evaluation and prediction.

---

## b. Dataset Description
The dataset used is the **Heart Disease Prediction Dataset** obtained from Kaggle.

- Dataset Type: Binary Classification
- Number of Instances: 1000+
- Number of Features: 13
- Target Variable: Presence of heart disease (0 = No, 1 = Yes)

The dataset contains medical attributes such as age, sex, cholesterol level, blood pressure, etc., used to predict heart disease.

---

## c. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.795122 | 0.878736 |	0.756303 | 0.873786 |	0.810811 | 0.597255 | 
| Decision Tree | 0.985366 | 0.985437 | 1.000000 | 0.970874 |0.985222 | 0.971151 |
| kNN | 0.834146 | 0.948553 | 0.800000	| 0.893204 | 0.844037 | 0.672727 |
| Naive Bayes | 0.800000 | 0.870550	| 0.754098 | 0.893204	| 0.817778	| 0.610224 |
| Random Forest (Ensemble) | 0.985366  |	1.000000  |	1.000000 | 0.970874 |	0.985222 | 0.971151 |
| XGBoost (Ensemble) | 0.985366 | 0.989435 | 1.000000 | 0.970874 | 0.985222	| 0.971151 |


---

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Achieved moderate performance with accuracy of 79.5% and AUC of 0.878. Shows good recall but lower precision compared to ensemble models, indicating it works well as a baseline model but struggles with complex relationships in data. |
| Decision Tree | Achieved very high accuracy (98.5%) and MCC (0.97), indicating excellent classification performance. However, decision trees may overfit the dataset due to their high variance. |
| kNN | Provided good performance with accuracy of 83.4% and high AUC score, but lower MCC compared to ensemble models. Performance depends heavily on feature scaling and is computationally expensive. |
| Naive Bayes | Produced moderate results with accuracy of 80%. Performs reasonably well but assumes feature independence, which may limit prediction accuracy in this dataset. |
| Random Forest (Ensemble)  | Achieved the best overall performance with perfect precision (1.0), highest AUC (1.0), and high MCC. Ensemble learning reduces overfitting and improves prediction stability. |
| XGBoost (Ensemble) | Performed similar to Random Forest with very high accuracy (98.5%) and strong MCC. Boosting technique improves prediction performance and handles complex patterns effectively. |


---

## Streamlit App
Live App Link: https://ml-classification-app-gfgvoaykschfbz3xbtrvxd.streamlit.app/

---

## GitHub Repository
Repository Link: https://github.com/VarthniCodes/ml-classification-streamlit
