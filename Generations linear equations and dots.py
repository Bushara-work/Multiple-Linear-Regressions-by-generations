import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy


# Load the dataset
df = pd.read_csv(r"C:\Users\busha\OneDrive\Desktop\Pokemon.csv")

# Define colors for each generation
generation_colors = {
    1: '#1f77b4',
    2: '#ff7f0e',
    3: '#2ca02c',
    4: '#d62728',
    5: '#9467bd',
    6: '#8c564b'
}

# Function to perform multiple linear regression for each generation
def multiple_linear_regression_by_generation(df):
    generations = df['Generation'].unique()
    models = {}
    for gen in generations:
        gen_df = df[df['Generation'] == gen]
        X = gen_df[['Defense']].values
        y = gen_df['Attack'].values
        model = LinearRegression()
        model.fit(X, y)
        models[gen] = model
    return models

# Function to plot multiple linear regression results by generation
def plot_multiple_regression_by_generation(df):
    models = multiple_linear_regression_by_generation(df)
    generations = df['Generation'].unique()

    plt.figure(figsize=(10, 6))
    for gen in generations:
        gen_df = df[df['Generation'] == gen]
        X = gen_df[['Defense']].values
        y = gen_df[['Attack']].values
        model = models[gen]
        predictions = model.predict(X)
        plt.plot(X, predictions, label=f'Generation {gen}', color=generation_colors[gen])

    for i, row in df.iterrows():
        color = generation_colors[row['Generation']]
        plt.scatter(row['Defense'], row['Attack'], color=color, edgecolor='black', linewidth=0.5)

    plt.xlabel('Defense')
    plt.ylabel('Attack')
    plt.title('Multiple Linear Regression by Generation')
    plt.legend()
    plt.show()

# Example usage
plot_multiple_regression_by_generation(df)
