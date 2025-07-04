# pip install pandas scikit-learn

import pandas as pd                                         # This imports Pandas, a Python library used to load and work with tabular data (like CSV files).
from sklearn.model_selection import train_test_split        # This imports a tool to split your data into Training and Test Dataset
from sklearn.naive_bayes import CategoricalNB               # This imports the Naïve Bayes classifier for categorical data
from sklearn.preprocessing import LabelEncoder              # This is used to convert text data into numbers.
from sklearn.metrics import accuracy_score                  # This tool is used to calculate the accuracy (how many predictions were correct).

# Load CSV data
data = pd.read_csv("weather.csv")

# Label encode each column
# Create a dictionary to store encoders for each column. 
# Why? So we can convert new input later and also decode predictions back to text.
encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    encoders[column] = le

# Split data into features (X) and target (y)
X = data.drop("Play", axis=1)               # X = everything except the column "Play"
y = data["Play"]                            # This is the target/output: Yes or No


# Split into training and test sets (e.g., 70% train, 30% test)
# X_train, y_train: 70% of data (used for training the model)
# X_test, y_test: 30% of data (used to test the model)
# random_state=0: Keeps the split same every time you run
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Create and train the Naive Bayes model
model = CategoricalNB()
model.fit(X_train, y_train)

# Predict on test set
# Use the trained model to predict the outcome (Yes or No) for the test data (X_test).
y_pred = model.predict(X_test)

# Decode predictions to original labels
# Convert predicted and actual results from numbers back to text using the saved LabelEncoder
predicted_labels = encoders["Play"].inverse_transform(y_pred)
actual_labels = encoders["Play"].inverse_transform(y_test)

# Show predictions
# Print each prediction and compare it with the actual answer.
# The function zip() is used to pair elements from two (or more) iterables together.
print("\nPredicted vs Actual:")
for pred, actual in zip(predicted_labels, actual_labels):
    print(f"Predicted: {pred}   Actual: {actual}")

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of Naive Bayes classifier: {accuracy * 100:.2f}%")