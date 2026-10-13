import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor


dataset = pd.read_csv(r"C:\Users\gopes\Downloads\Project DS\SE.csv")

X = dataset.iloc[:, 0:1].values
Y = dataset.iloc[:, 1].values

# -------------------------------
# CREATE SGD MODEL
# -------------------------------

# Create an object of the Stochastic Gradient Descent Regressor
model = SGDRegressor(

    # Maximum number of iterations (epochs) for training
    max_iter=1000,

    # Training stops when improvement is smaller than 0.001
    tol=1e-3,

    # Use a constant learning rate during training
    learning_rate='constant',

    # Learning rate = 0.01
    # It controls how much the weight changes in each update
    eta0=0.01,

    # Fixes the random behavior so we get reproducible results
    random_state=42
)


# Train the SGD model using input X and output y
model.fit(X, Y)


# Predict marks for a student who studied 6 hours
prediction = model.predict([[1.8]])


# -------------------------------
# DISPLAY RESULTS
# -------------------------------

# Print the predicted marks
print("Predicted Salary:", prediction[0])

# Print the learned weight/coefficient (m)
# Formula: y = mx + c
print("Weight (m):", model.coef_[0])

# Print the learned intercept/constant (c)
# Formula: y = mx + c
print("Intercept (c):", model.intercept_[0])

