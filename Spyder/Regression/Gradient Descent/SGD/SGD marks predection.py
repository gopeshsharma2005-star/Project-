import numpy as np
from sklearn.linear_model import SGDRegressor

# X represents the independent variable (input feature)
# Here, it represents Hours Studied
X = np.array([[1], [2], [3], [4], [5]])

# y represents the dependent variable (target/output)
# Here, it represents Marks obtained
y = np.array([10, 20, 30, 40, 50])


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
model.fit(X, y)


# Predict marks for a student who studied 6 hours
prediction = model.predict([[6]])


# -------------------------------
# DISPLAY RESULTS
# -------------------------------

# Print the predicted marks
print("Predicted Marks:", prediction[0])

# Print the learned weight/coefficient (m)
# Formula: y = mx + c
print("Weight (m):", model.coef_[0])

# Print the learned intercept/constant (c)
# Formula: y = mx + c
print("Intercept (c):", model.intercept_[0])
