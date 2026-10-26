# Import required libraries
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score

# Create a simple dataset
# X represents study hours
X = np.array([[1], [2], [3], [4], [5], [6]])

# y represents the result: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1])

# Create the machine learning model
model = DecisionTreeClassifier(random_state=0)

# Create the LOOCV object
loo = LeaveOneOut()

# Train and evaluate the model, leaving one row out each time
scores = cross_val_score(
    model,
    X,
    y,
    cv=loo,
    scoring="accuracy"
)

# Display the accuracy of each fold
print("Accuracy of each fold:", scores)

# Display the total number of folds
print("Total folds:", len(scores))

# Calculate and display average accuracy
print("Average LOOCV Accuracy:", scores.mean())

# Convert accuracy into percentage
print("Accuracy Percentage:", scores.mean() * 100, "%")


