# Import required library
import pandas as pd

# Load the dataset into DataFrame
Development = pd.read_csv("HDI.csv")

# Display first five rows
print("First 5 Rows:")
print(Development.head())

# Show dataset dimensions
print("\nDataset Shape:")
print(Development.shape)

# Display column names
print("\nColumn Names:")
print(Development.columns)

# Show dataset information
print("\nDataset Information:")
print(Development.info())

# Display statistical summary
print("\nStatistical Summary:")
print(Development.describe())