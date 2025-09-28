# Teen Phone Addiction Prediction using Machine Learning  
A machine learning project to classify and analyze phone/technology addiction in teenagers based on survey data.  

## Table of Contents  
- [About the Project](#about-the-project)  
- [Dataset](#dataset)  
- [Features / Variables](#features--variables)  
- [Methodology](#methodology)  
- [Requirements](#requirements)  

## About the Project  
This project aims to build a classifier that predicts teen phone addiction based on certain features (from a survey dataset). It includes data cleaning, feature selection, model training, evaluation, and a simple interface (via Streamlit) to test predictions.

## Dataset  
- **teen_phone_addiction_dataset.csv** — the original dataset  
- **teen_phone_addiction_binary.csv** — processed/cleaned version with binary labels  
These datasets contain responses from to questions about their phone usage, habits, and demographics.

## Features / Variables  
Some of the features included are (you can list actual ones here):  
- Daily usage hours  
- Frequency of checking phone  
- Sleep disturbance  
- Social interactions  
- Demographic features (age, gender, etc.)  

## Methodology  
1. **Data Preprocessing** — cleaning missing values, encoding categories, balancing classes  
2. **Feature Selection / Engineering** — selecting important predictors  
3. **Modeling** — applying classification algorithms (e.g. Logistic Regression, Random Forest, SVM)  
4. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrix  
5. **Deployment / Interface** — a simple front-end to input new data and see predictions  

### Requirements  
You can install required libraries via pip:

```bash
pip install -r requirements.txt
