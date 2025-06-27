# pip install pandas matplotlib scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression               # This imports the LinearRegression model from scikit-learn to perform regression analysis.
from sklearn.metrics import mean_squared_error                  # This imports a function to calculate Mean Squared Error — a measure of how accurate your regression is.

# Step 1: Load dataset
data = pd.read_csv("data.csv")

# Step 2: Split into input (X) and output (y)
# Note: double brackets [[ ]] are used to keep it as a column table (2D), not just a list.
X = data[["Experience"]]  # X contains the input feature — here, it's Experience. 
y = data["Salary"]        # target

print(y)
# Step 3: Train the regression model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict the values
y_pred = model.predict(X)

# Step 5: Print Linear Regression  equation and error
# Salary=slope×Experience+intercept === mX + b
print(f"Regression Equation: Salary = {model.coef_[0]:.2f} * Experience + {model.intercept_:.2f}")
print(f"Mean Squared Error: {mean_squared_error(y, y_pred):.2f}")

# Step 6: Plot Regression Line
plt.figure(figsize=(10, 5))

# Actual vs Predicted plot
plt.subplot(1, 2, 1)                                    # Prepare the left half of the screen for the first plot (1 row, 2 columns, 1st plot).
plt.scatter(X, y, color='blue', label='Actual Salary')
plt.plot(X, y_pred, color='red', label='Predicted Line')
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.title("Linear Regression")
plt.legend()

# Step 7: Residual(errors) Plot 
plt.subplot(1, 2, 2)                                    # Prepare the right half of the screen for the 2nd plot (1 row, 2 columns, 2nd plot).
residuals = y - y_pred
plt.scatter(X, residuals, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Experience (Years)")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.tight_layout()
plt.show()