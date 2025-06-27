# pip install pandas scikit-learn matplotlib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree    # TO build a decision tree model and to draw the decision tree diagram
import matplotlib.pyplot as plt                               # For VIsualization
from sklearn.preprocessing import LabelEncoder                # used to convert text (like "Sunny", "Cool") into numbers (like 0, 1), which the model can understand.

# Load the dataset from CSV file using pandas
data = pd.read_csv('play_tennis.csv')

# Encode or Convert the column values (e.g., Sunny, Rainy) into numbers (0, 1, 2, …) with separate LabelEncoder 
# Replace the original column with encoded numbers
label_encoders = {}
for column in data.columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le  # Save encoder for later use

# Now your data is all numbers, ready for machine learning. Check the data by printing it's values
# The header (column name) like 'Outlook' is never changed or encoded — only the values below the header row are processed.
print(data)

# Split features and target
X = data.drop('Play', axis=1)       # X will contain all input columns (Outlook, Temperature, etc.), but not 'Play'.
y = data['Play']                    # y contains only the target/output column (Play) → this is what we want to predict.


print(X)
print(y)

# Train decision tree using ID3 (entropy)
# Create a decision tree classifier using the ID3 algorithm.
model = DecisionTreeClassifier(criterion='entropy')           # criterion='entropy' means we are using Information Gain (ID3 method) to split the tree.
model.fit(X, y)                                               # Train (fit) the model using the features (X) and target (y). Now the tree has learned the rules from your data.


# Visualize the tree
plt.figure(figsize=(10, 6))               # Set the size of the plot (graph) to 10 by 6.
plot_tree(model, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)     # Show feature names, Show class names (No and Yes), and Fill colors to make it easier to read
plt.title("Decision Tree (ID3) - Play Tennis")
plt.show()


# Test the Trained Model or Predict a New Sample
# Input sample to classify: Outlook=Sunny, Temperature=Cool, Humidity=High, Wind=Strong
# Encode using correct label encoders
sample_dict = {
    "Outlook": "Sunny",
    "Temperature": "Cool",
    "Humidity": "High",
    "Wind": "Strong"
}

# You take each value (like "Sunny") and convert it to number using the same encoder used during training.
# 2D List
sample_encoded = [[
    label_encoders["Outlook"].transform([sample_dict["Outlook"]])[0],
    label_encoders["Temperature"].transform([sample_dict["Temperature"]])[0],
    label_encoders["Humidity"].transform([sample_dict["Humidity"]])[0],
    label_encoders["Wind"].transform([sample_dict["Wind"]])[0],
]]

# Predict the result
# Ask the model to predict if we can play tennis with the new weather conditions.
prediction = model.predict(sample_encoded)

# Convert the numeric prediction back to "Yes" or "No" using the saved encoder.
result = label_encoders["Play"].inverse_transform(prediction)

print(f"\nPrediction for the sample {sample_dict}: {result[0]}")