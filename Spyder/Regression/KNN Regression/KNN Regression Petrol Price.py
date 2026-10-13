# Importing the libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline

# Importing the dataset
dataset = pd.read_csv(
    r"C:\Users\gopes\Downloads\Project DS\Petrol_Price_1_Year_Dataset.csv"
)

# Removing extra spaces from column names
dataset.columns = dataset.columns.str.strip()

# Display column names
print("Dataset Columns:", dataset.columns.tolist())

# Selecting X and Y
# X = Crude Oil Price
# Y = Petrol Price
X = dataset.iloc[:, 1:2].apply(pd.to_numeric, errors='coerce')
Y = pd.to_numeric(dataset.iloc[:, 2], errors='coerce')

# Removing rows where Y is missing
valid_rows = Y.notna()
X = X.loc[valid_rows]
Y = Y.loc[valid_rows]

# Creating the KNN Regression model
model = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    KNeighborsRegressor(
        n_neighbors=3,
        weights='distance'
    )
)

# Training the model
model.fit(X, Y)

# Predicting petrol price for crude oil price = 70
new_data = pd.DataFrame({
    X.columns[0]: [70]
})

prediction = model.predict(new_data)

# Displaying the results
print("\nCrude Oil Price:", 70, "USD/barrel")
print("Predicted Petrol Price:", round(prediction[0], 2), "per litre")
