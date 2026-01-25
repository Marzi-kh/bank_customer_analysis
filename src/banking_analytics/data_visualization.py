import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.banking_analytics import load_file


df = load_file.load_data()

def plot_age_distribution(df):
    """plot age distribution"""
    plt.figure()
    sns.histplot(df['age'], bins = 30 , kde=True)
    plt.title('Age Distribution of customers')
    plt.xlabel('Age')
    plt.ylabel('Counts')
    plt.show()

def plot_job_distribution(df):
    """plot job distribution"""
    plt.figure()
    sns.countplot(x = 'job', data = df)
    plt.xticks(rotation = 45)
    plt.title('Job Distribution of customers')
    plt.xlabel('Job')
    plt.ylabel('Counts')
    plt.show()

def plot_correlation_heatmap(df):
    """plot correlation heatmap"""
    numeric_df = df.select_dtypes(include =["int64" , "float64"] )
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.xticks(rotation = 45)
    plt.title('Correlation Heatmap')
    plt.show()

def main():
    file_path = "/Users/skhansar19/PycharmProjects/bank_customer_analysis/data/bank.csv"

    plot_age_distribution(df)
    plot_job_distribution(df)
    plot_correlation_heatmap(df)

if __name__ == "__main__":
    main()