Bank Customer Analysis

This project explores a bank marketing dataset and builds a simple machine learning model to 
predict whether a customer will subscribe to a term deposit. The goal is to understand which 
customer characteristics are associated with higher subscription rates and to create a b
aseline predictive model.

The project includes data exploration, feature engineering, and a logistic regression model 
implemented using a scikit-learn pipeline.


# Dataset

The dataset contains 11,162 customer records with 17 variables describing demographic information,
financial status, and previous marketing interactions.

The target variable is deposit, which indicates whether the customer subscribed to a term deposit.

Target distribution:

No: 5,873 customers (52.6%)
Yes: 5,289 customers (47.4%)

Data quality checks show that the dataset is clean:

No missing values
No duplicate rows


Exploratory Data Analysis

The analysis begins with basic dataset inspection (head, summary statistics, missing values, 
column types).  
I also implemented a quick EDA function that summarizes how different customer characteristics 
relate to the probability of subscribing.

Some interesting patterns appear in the data.

Job

Student → 74.7% subscription rate  
Retired → 66.3%  
Management → 50.7%  
Blue-collar → 36.4%

Marital status

Single → 54.3%  
Married → 43.4%

Education

Tertiary education → 54.1%  
Primary education → 39.4%

Housing loan

No housing loan → 57.0%  
Housing loan → 36.6%

Previous campaign outcome

Previous success → 91.3%  
Unknown outcome → 40.7%

These patterns suggest that job type, housing status, and previous campaign success may be 
important predictors.



Numerical Feature Comparison

I also compared the mean values of numerical features for customers who subscribed and those who 
did not.

Some differences stand out:

Customers who subscribed tend to have higher account balances
They have more previous contacts
They often have longer call durations

However, duration is a special case and was removed from the model to prevent data leakage (see below).

Feature Engineering

The dataset uses the value 999 in pdays to indicate that the customer had not been contacted previously.

To make this easier for the model to interpret:

created a binary feature  pdays_was_contacted
replaced pdays = 999 with -1

This helps distinguish customers who had previous contact from those who did not.


Preventing Data Leakage

The variable duration was removed from model training.

Although call duration is strongly correlated with the target, it represents the length of 
the marketing call itself. In practice, this value would only be known after the call finishes,
so including it would introduce target leakage.

For that reason, it was excluded from the baseline model.


Modeling Approach

A Logistic Regression model was used as the baseline classifier.

The preprocessing pipeline includes:

StandardScaler** for numerical features
OneHotEncoder** for categorical features

The dataset was split into:

80% training data
20% test data

The split was stratified to preserve the class distribution.

Model Results

CONFUSION MATRIX
[[958 217]
 [463 595]]


CLASSIFICATION REPORT
              precision    recall  f1-score   
           0       0.67      0.82      0.74      
           1       0.73      0.56      0.64      

Accuracy: 70%

ROC-AUC: 0.758

The baseline model performs reasonably well for a first attempt. It correctly identifies many 
non-subscribing customers, but recall for the positive class could still be improved.



Output Files

The model generates two useful output files.

top_customers.csv

This file contains customers with the highest predicted probability of subscribing. 
It can be used to identify promising leads for targeted marketing campaigns.

segment_summary.csv

This file summarizes predicted and actual subscription rates across customer segments such as:

job
age group
marital status

This helps connect model predictions with interpretable business insights.



Project Structure
bank_customer_analysis
│
├── data
├── outputs
├── src
│ ├── load_file.py
│ ├── data_exploration.py
│ ├── data_visualization.py
│ ├── data_cleaning.py
│ ├── train_baseline.py
│
└── README.md


How to Run

You can run the full workflow using the pipeline script.

python -m src.pipeline

This script runs the main steps of the project:

1. Loads the dataset
2. Handles missing values
3. Creates new features (such as age groups)
4. Trains the baseline logistic regression model
5. Saves the results to the outputs/ folder

The script will generate the following files:

top_customers.csv — customers with the highest predicted probability of subscribing
segment_summary.csv — summary of predicted and actual subscription rates across customer segments

You can also run individual parts of the project:

Data exploration: python -m src.data_exploration

Data cleaning: python -m src.data_cleaning

Visualization: python -m src.data_visualization

Model only: python -m src.train_baseline

Final Thoughts

This project demonstrates a simple end-to-end workflow for a classification problem:

exploring and understanding the dataset  
identifying useful patterns through quick EDA summaries  
performing simple feature engineering  
building a baseline predictive model  
translating predictions into interpretable business insights

Even with a relatively simple model like logistic regression, the results already provide useful 
signals for marketing targeting. By identifying customer segments with higher predicted 
probabilities of subscribing, marketing efforts could potentially focus on the most promising leads.

Future improvements could include trying more advanced models, tuning hyperparameters, and 
exploring additional feature engineering to improve prediction performance.