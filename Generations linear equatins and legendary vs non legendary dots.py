import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv(r"C:\Users\busha\OneDrive\Desktop\Pokemon.csv")

# Define colors for each Pokémon type
type_colors = {
    'Grass': '#78C850',
    'Poison': '#A040A0',
    'Fire': '#F08030',
    'Flying': '#A890F0',
    'Water': '#6890F0',
    'Bug': '#A8B820',
    'Normal': '#A8A878',
    'Electric': '#F8D030',
    'Ground': '#E0C068',
    'Fairy': '#EE99AC',
    'Fighting': '#C03028',
    'Psychic': '#F85888',
    'Rock': '#B8A038',
    'Steel': '#B8B8D0',
    'Ice': '#98D8D8',
    'Ghost': '#705898',
    'Dragon': '#7038F8',
    'Dark': '#705848'
}

# Define colors for legendary status
legendary_colors = {
    True: 'gold',
    False: 'silver'
}

# Function to get color for a Pokémon based on its type(s)
def get_color(row):
    type = row['Legendary']
    return legendary_colors[type]

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
        y = gen_df['Attack'].values
        model = models[gen]
        predictions = model.predict(X)
        plt.plot(X, predictions, label=f'Generation {gen}')

    for i, row in df.iterrows():
        color = get_color(row)
        plt.scatter(row['Defense'], row['Attack'], color=legendary_colors[row['Legendary']], edgecolor=color, linewidth=2)

    plt.xlabel('Defense')
    plt.ylabel('Attack')
    plt.title('Multiple Linear Regression by Generation')
    plt.legend()
    plt.show()

# Example usage
plot_multiple_regression_by_generation(df)
