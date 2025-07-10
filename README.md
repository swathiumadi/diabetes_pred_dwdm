# diabetes_pred_dwdm
A Data Warehousing and Data Mining (DWDM) project for predicting diabetes using the Pima Indians dataset with machine learning algorithms and preprocessing techniques.
#  Diabetes Prediction using Machine Learning 

This project was developed as part of the **Data Warehousing and Data Mining (DWDM)** course to predict whether a person has diabetes based on medical attributes. The dataset used is the **PIMA Indians Diabetes Dataset**, a widely used benchmark in medical ML applications.

---

##  Project Structure 

- `preprocessing.ipynb`: Data cleaning, handling missing values, normalization, PCA, outlier removal, and feature analysis.
- `modeling.ipynb`: Applied ML models like Decision Tree, KNN, Logistic Regression, Random Forest, Naive Bayes with hyperparameter tuning.
- `preprocessed_data.csv`: Cleaned version of the original dataset used for modeling.
- `diabetes.csv`: Original dataset containing the `Outcome` label.

---

## Dataset 

- Source: [PIMA Indian Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
- Target: Outcome (0 = No Diabetes, 1 = Diabetes)

---

## Machine Learning Models Used 

- Decision Tree Classifier (with GridSearchCV)
- K-Nearest Neighbors (with scaling & tuning)
- Logistic Regression
- Random Forest (with both GridSearchCV and RandomizedSearchCV)
- Naive Bayes

---

##  Model Performance Comparison 

| Model                      | Accuracy (%) |
|----------------------------|--------------|
| Decision Tree              | 70           |
| K-Nearest Neighbors (KNN)  | 73           |
| Logistic Regression        | 75           |
| Random Forest              | 75           |
| Naive Bayes                | 75           |

> Logistic Regression, Random Forest, and Naive Bayes showed the best performance at 75% accuracy.  
>  Random Forest and KNN were fine-tuned using cross-validation methods.   
>  This comparative analysis demonstrates how model selection and tuning impact prediction performance. 

---

##  Requirements 

- Python 3.x
- pandas
- scikit-learn
- matplotlib / seaborn (for optional visualizations)
- numpy

---
