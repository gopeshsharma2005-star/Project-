import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Creating the dataset
dataset = pd.DataFrame({
    "House_Size": [800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700],
    "House_Price": [32, 36, 40, 44, 48, 52, 56, 60, 64, 68]
})

# Displaying the dataset
print("Dataset:")
print(dataset)

# Selecting X and Y
X = dataset[['House_Size']]
Y = dataset['House_Price']

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=5
)

# Creating the KNN Regression model
model = make_pipeline(
    StandardScaler(),
    KNeighborsRegressor(
        n_neighbors=3,
        weights='distance'
    )
)

# Training the model
model.fit(X_train, Y_train)

# Predicting prices for test data
Y_pred = model.predict(X_test)

# Evaluating the model
print("\nMean Absolute Error:",
      mean_absolute_error(Y_test, Y_pred))

print("Mean Squared Error:",
      mean_squared_error(Y_test, Y_pred))

print("Root Mean Squared Error:",
      np.sqrt(mean_squared_error(Y_test, Y_pred)))

print("R2 Score:", r2_score(Y_test, Y_pred))

# Predicting price for a house of size 1800
new_data = pd.DataFrame({
    'House_Size': [1450]
})

prediction = model.predict(new_data)

# Displaying the result
print("\nHouse Size:", 1450)
print("Predicted House Price:", round(prediction[0], 2))
