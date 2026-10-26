# Import required libraries
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score

# Create a dataset containing study hours
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

# 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Create the machine learning model
model = DecisionTreeClassifier(random_state=0)

# Divide the dataset into 5 folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation and calculate accuracy
scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="accuracy"
)

# Display the accuracy of each fold
print("Accuracy of each fold:", scores)

# Display the number of folds
print("Total folds:", len(scores))

# Calculate average accuracy
print("Average Accuracy:", scores.mean())

# Display accuracy as a percentage
print("Accuracy Percentage:", scores.mean() * 100, "%")
