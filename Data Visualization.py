# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
Development = pd.read_csv("HDI.csv")

# Select first 20 rows to avoid overcrowding
data1 = Development.iloc[:20]


# 1. Display Unique Country Values


print("Unique Countries:")
print(Development["Country"].unique())


# 2. Mean Years of Schooling vs HDI


plt.figure(figsize=(10,6))

sns.stripplot(
    x="Mean years of schooling",
    y="HDI",
    data=data1,
    jitter=True
)

plt.xticks(rotation=90)

plt.title("Mean Years of Schooling vs HDI")

plt.show()

# 3. Life Expectancy vs HDI

plt.figure(figsize=(10,6))

sns.stripplot(
    x="Life expectancy",
    y="HDI",
    data=data1,
    jitter=True
)

plt.xticks(rotation=90)

plt.title("Life Expectancy vs HDI")

plt.show()

# 4. Correlation Heatmap

heat = Development.iloc[:,[0,1,2,3,4,5,6,7]]

plt.figure(figsize=(10,8))

sns.heatmap(
    heat.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()