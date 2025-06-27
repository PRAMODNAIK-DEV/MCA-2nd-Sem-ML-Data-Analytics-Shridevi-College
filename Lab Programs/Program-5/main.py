# Program-5: Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set. Print both correct and wrong predictions. 

# Install sklearn using - pip install scikit-learn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data          # List: iris dataset has 150 Rows with 4 Features: Sepal Length, Sepal Width, Petal Length, Petal Width
y = iris.target        # List: Target classes: 0 -> Iris Setosa, 1 -> Iris Versicolor, 2 -> Iris Virginica

print(len(X))
print(y)

# Split the dataset into 70% training and 30% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# test_size=0.3: This means 30% of the data will be used for testing, and the remaining 70% will be used for training.
# random_state=42: This is used to control the randomness of how the data is split. The number 42 is just a number — you could use 0, 1, 99, or any other integer.


# Create the k-NN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Predict using the trained model
y_pred = knn.predict(X_test)

# Print accuracy
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

# Compare predictions
print("🔍 Prediction Results:")
for i in range(len(y_test)):
    actual = iris.target_names[y_test[i]]
    predicted = iris.target_names[y_pred[i]]
    status = "✅ Correct" if y_test[i] == y_pred[i] else "Wrong"
    print(f"Sample {i + 1}: Predicted = {predicted}, Actual = {actual} -> {status}")
