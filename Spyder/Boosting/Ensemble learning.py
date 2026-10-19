import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Create a sample dataset
data = {
    "StudyHours": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 2,
        3, 4, 5, 6, 7, 8, 9, 1, 5, 7,
        2, 4, 6, 8, 3, 9, 5, 6, 1, 8
    ],
    "Attendance": [
        30, 40, 50, 60, 70, 80, 90, 95, 98, 35,
        45, 65, 75, 85, 92, 96, 99, 25, 55, 88,
        42, 62, 82, 94, 48, 97, 72, 78, 20, 93
    ],
    "Pass": [
        0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0, 0, 1,
        0, 0, 1, 1, 0, 1, 1, 1, 0, 1
    ]
}

df = pd.DataFrame(data)

# 2. Separate input and output
X = df[["StudyHours", "Attendance"]]
y = df["Pass"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 4. Create the base models
decision_tree = DecisionTreeClassifier(
    max_depth=2,
    random_state=42
)

logistic = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=3)
)

# 5. BAGGING
bagging_model = BaggingClassifier(
    estimator=decision_tree,
    n_estimators=10,
    random_state=42
)

# 6. BOOSTING: AdaBoost
boosting_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=10,
    random_state=42
)

# 7. STACKING
stacking_model = StackingClassifier(
    estimators=[
        ("tree", decision_tree),
        ("logistic", logistic),
        ("knn", knn)
    ],
    final_estimator=LogisticRegression(),
    cv=3
)

# 8. Train and evaluate all models
models = {
    "Bagging": bagging_model,
    "AdaBoost": boosting_model,
    "Stacking": stacking_model
}

for name, model in models.items():

    # Train the model
    model.fit(X_train, y_train)

    # Predict the test data
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Display results
    print("\nAlgorithm:", name)
    print("Actual values:", y_test.tolist())
    print("Predicted values:", y_pred.tolist())
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# 9. Predict for a new student using Stacking
new_student = pd.DataFrame({
    "StudyHours": [6],
    "Attendance": [85]
})

prediction = stacking_model.predict(new_student)

if prediction[0] == 1:
    print("\nNew Student Prediction: PASS")
else:
    print("\nNew Student Prediction: FAIL")
