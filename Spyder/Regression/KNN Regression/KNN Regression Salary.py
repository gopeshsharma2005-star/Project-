
# Importing the libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

# Importing the dataset
dataset = pd.read_csv(
    r"C:\Users\gopes\Downloads\Project DS\SE.csv"
)

# Selecting X and Y
X = dataset[['YearsExperience']]
Y = dataset['Salary']

# Splitting the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
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

# Predicting salaries for test data
Y_pred = model.predict(X_test)

# Evaluating the model
print("Mean Absolute Error:", mean_absolute_error(Y_test, Y_pred))
print("Mean Squared Error:", mean_squared_error(Y_test, Y_pred))
print("Root Mean Squared Error:",
      np.sqrt(mean_squared_error(Y_test, Y_pred)))
print("R2 Score:", r2_score(Y_test, Y_pred))

# Predicting salary for a new experience value
new_data = pd.DataFrame({
    'YearsExperience': [1.8]
})

prediction = model.predict(new_data)

# Displaying the results
print("\nYears of Experience:", new_data.iloc[0, 0])
print("Predicted Salary:", round(prediction[0], 2))
