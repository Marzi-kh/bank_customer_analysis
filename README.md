# Bank Customer Marketing Analysis

## Project Overview

In this project, I worked with a bank marketing dataset to understand customer behavior and 
build a simple model to predict whether a customer will subscribe to a term deposit.

The main idea was to go through a full workflow — from exploring the data to building a 
baseline model — and see how the results could actually be used in a business context.

## Dataset

The dataset contains information about bank customers, including:

- Age, job, marital status, education  
- Financial details like balance, loans, and housing  
- Interaction history from previous marketing campaigns  

The target variable is:

- deposit — whether the customer subscribed (yes/no)

## Data Exploration

The dataset has:

- 11,162 rows  
- 17 columns  

I checked for basic data quality issues and found no missing values or duplicates.

Some things that stood out:

- The target is fairly balanced (~53% no, ~47% yes), which is nice for modeling  
- pdays = -1 means the customer was never contacted before  
- balance varies a lot and even includes negative values  
- previous campaign outcome (poutcome) looks like an important feature  

Categorical variables like job or education also seem useful for segmenting customers.

## Data Preparation

Since the data was already quite clean, I didn’t need heavy preprocessing.

What I did:

- Verified there are no missing values or duplicates  
- Created a simple feature called age_group (Young, Adult, Middle_age, Senior)  
- Left categorical variables as they are and handled encoding inside the model pipeline  

## Model

I used a logistic regression model as a baseline.

The pipeline includes:

- Scaling numerical features  
- One-hot encoding categorical features  
- Training the model  

The data was split into 80% training and 20% testing.

## Results

- Accuracy: 83%  
- ROC-AUC: 0.91  

The model does a good job separating customers who are likely to subscribe from those who 
are not.

From a practical point of view:

- It captures most of the positive cases  
- It doesn’t produce too many false positives  

## Business Output

To make the results more useful:

- top_customers.csv  
  Customers ranked by predicted probability  
  → This can be used to target high-potential customers first  

- segment_summary.csv  
  Aggregated performance by groups (job, age group, marital status)  
  → This helps identify which segments are most likely to respond  

## Visualization

Basic visualizations are generated, including:

- Age distribution  
- Job distribution  
- Correlation heatmap  

Plots are saved in:

outputs/figures/

## Project Structure

src/
├── load_file.py
├── data_exploration.py
├── data_cleaning.py
├── data_visualization.py
├── train_baseline.py

data/
outputs/
README.md
requirements.txt

## Installation

pip install -r requirements.txt

## How to Run

python -m src.pipeline

This runs the full workflow: loading the data, preparing it, training the model, and saving 
the results to the outputs/ folder.
## Other Scripts

You can also run individual parts of the project:

Data exploration:
python -m src.data_exploration

Data cleaning:
python -m src.data_cleaning

Visualization:
python -m src.data_visualization

Model only:
python -m src.train_baseline

## Final Thoughts

This project demonstrates a simple and practical workflow:

- Exploring and understanding data  
- Building a baseline predictive model  
- Translating predictions into actionable business insights  

Even with a simple model, meaningful improvements in marketing targeting can be achieved.