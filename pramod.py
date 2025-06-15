import pandas as pd

# Load dataset
df = pd.read_csv("sales.csv")

# Check top rows
print("Head:\n", df.head())

# Drop missing values
df = df.dropna()

# Filter: Get sales above 5000
filtered_df = df[df['Sales'] > 5000]

# Add new column: profit margin
df['Profit Margin'] = df['Profit'] / df['Sales']

print(df)
# Group by Region
region_sales = df.groupby('Region')['Sales'].sum()
print("\nSales by Region:\n", region_sales)

import matplotlib.pyplot as plt

# Bar chart for region sales
region_sales.plot(kind='bar', color='skyblue')
plt.title('Sales by Region')
plt.xlabel('Region')
plt.ylabel('Sales')
plt.tight_layout()
plt.show()
