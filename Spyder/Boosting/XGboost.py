# Import pandas to create and manage the dataset
import pandas as pd

# Import XGBoost classification algorithm
from xgboost import XGBClassifier

# Split the dataset into training and testing data
from sklearn.model_selection import train_test_split

# Import accuracy and confusion matrix metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create a sample dataset for students
data = {
    "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4,
                   6, 7, 3, 5, 8, 9, 1, 3, 6, 9],
    "Attendance": [40, 45, 50, 55, 60, 65, 70, 85, 42, 58,
                   68, 75, 48, 62, 88, 95, 35, 52, 72, 98],
    # 0 = Fail, 1 = Pass
    "Result": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0,
               1, 1, 0, 1, 1, 1, 0, 0, 1, 1]
}

# Convert the dictionary into a DataFrame
df = pd.DataFrame(data)

# Select input features (independent variables)
X = df[["StudyHours", "Attendance"]]

# Select the target variable (dependent variable)
y = df["Result"]

# Split the data: 80% training and 20% testing
# random_state=42 makes the split reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Create the XGBoost classification model
model = XGBClassifier(
    n_estimators=50,       # Number of boosting trees
    max_depth=3,           # Maximum depth of each tree
    learning_rate=0.1,     # Controls each tree's contribution
    objective="binary:logistic",  # Binary classification
    eval_metric="logloss", # Metric used during training
    random_state=42
    # Log loss measures how well the predicted probabilities match the actual classes.
)

# Train the model on training data
model.fit(X_train, y_train)

# Predict pass/fail results for unseen test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Display the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display precision, recall, and F1-score
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Predict the result for a new student
# New student: 6 study hours and 75% attendance
new_student = pd.DataFrame({
    "StudyHours": [6],
    "Attendance": [75]
})

# Predict the student's class
prediction = model.predict(new_student)[0]

# Convert the numerical prediction into readable output
if prediction == 1:
    print("\nPredicted Result: Pass")
else:
    print("\nPredicted Result: Fail")
