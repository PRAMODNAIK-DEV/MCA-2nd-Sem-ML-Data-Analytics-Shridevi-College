import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Load CSV data
data = pd.read_csv("dataset.csv")

# Step 2: Convert to numerical array
X = data.values  # or data[['Height', 'Weight']].values if more columns

# Step 3: Apply K-Means clustering
k = int(input("Enter the number of Clusters (k): "))  # You can change number of clusters
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)

# Step 4: Get cluster labels
labels = kmeans.labels_
print("\nCluster Labels:", labels)

# Step 5: Add labels to data and show result
data['Cluster'] = labels
print("\nClustered Data:\n", data)

# Step 6: Plot the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', label='Centroids', marker='X')
plt.title('K-Means Clustering')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()
plt.grid(True)
plt.show()
