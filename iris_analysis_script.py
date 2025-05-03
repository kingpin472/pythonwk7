import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Simulate a missing value for demonstration
iris_df.loc[0, 'sepal length (cm)'] = np.nan

# Clean missing data: Fill with column mean
iris_df.fillna(iris_df.mean(numeric_only=True), inplace=True)

# Task 1: Explore the dataset
print("First 5 rows:")
print(iris_df.head())

print("\nData types:")
print(iris_df.dtypes)

print("\nMissing values:")
print(iris_df.isnull().sum())

# Task 2: Basic Statistics
print("\nDescriptive statistics:")
print(iris_df.describe())

print("\nAverage values grouped by species:")
print(iris_df.groupby('species').mean())

# Task 3: Visualizations
sns.set(style="whitegrid")

# Line plot (just to visualize all rows as trends)
iris_df.drop(columns='species').plot(title="Feature Trends Over Samples", figsize=(10, 5))
plt.xlabel("Sample Index")
plt.ylabel("Measurement (cm)")
plt.tight_layout()
plt.show()

# Bar chart: Average petal length per species
iris_df.groupby("species")["petal length (cm)"].mean().plot(kind='bar', title="Average Petal Length by Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# Histogram: Sepal Width
iris_df["sepal width (cm)"].plot(kind='hist', bins=20, title="Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.tight_layout()
plt.show()

# Scatter plot: Sepal Length vs Petal Length
sns.scatterplot(data=iris_df, x="sepal length (cm)", y="petal length (cm)", hue="species")
plt.title("Sepal Length vs Petal Length by Species")
plt.tight_layout()
plt.show()
