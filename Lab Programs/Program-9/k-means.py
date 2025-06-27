# Install the dependencies
# pip install pandas matplotlib scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load CSV data
file_path = "dataset.csv" 
data = pd.read_csv(file_path)


# Drop non-numeric columns if any (optional, based on CSV structure)
# numeric_data = data.select_dtypes(include=["float64", "int64"])

# Ask user for number of clusters
k = int(input("Enter the number of clusters (k): "))

# Create KMeans model
kmeans = KMeans(n_clusters=k, random_state=0)

# Fit the model to the data
kmeans.fit(data)

# Add cluster labels to the data
data['Cluster'] = kmeans.labels_

# Print cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Print data with cluster labels
print("\nData with Cluster Labels:")
print(data)

# Plotting (only if 2 features)
if numeric_data.shape[1] == 2:
    plt.scatter(numeric_data.iloc[:, 0], numeric_data.iloc[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='X', label='Centroids')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.title('K-Means Clustering')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("\nPlotting is only supported for 2D data.")

