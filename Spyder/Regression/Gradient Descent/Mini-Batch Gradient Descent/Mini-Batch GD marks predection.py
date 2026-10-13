import numpy as np

# x = Hours Studied
x = np.array([1, 2, 3, 4, 5, 6])

# y = Marks
y = np.array([10, 20, 30, 40, 50, 60])

# m = Slope
m = 0

# c = Constant / Intercept
c = 0

# Learning rate
learning_rate = 0.01

# Number of training rounds
training_rounds = 10

# We will use 2 records in each batch
batch_size = 2


# -----------------------------------------
# 3. Mini-Batch Gradient Descent
# -----------------------------------------

for round_no in range(training_rounds):

    print("\nTraining Round:", round_no + 1)

    # Divide data into batches of 2
    for i in range(0, len(x), batch_size):

        # Select 2 x values
        x_batch = x[i:i + batch_size]

        # Select corresponding 2 y values
        y_batch = y[i:i + batch_size]


        print("Batch X:", x_batch)
        print("Batch Y:", y_batch)



        y_pred = m * x_batch + c

        # -----------------------------------------
        # 5. Calculate Error
        # -----------------------------------------

        error = y_batch - y_pred

        # -----------------------------------------
        # 6. Calculate Gradient
        # -----------------------------------------

        # Gradient for slope
        dm = -2 * np.mean(x_batch * error)

        # Gradient for constant
        dc = -2 * np.mean(error)

        # -----------------------------------------
        # 7. Update m and c
        # -----------------------------------------

        m = m - learning_rate * dm

        c = c - learning_rate * dc

        print("m (Slope):", m)
        print("c (Constant):", c)


# -----------------------------------------
# 8. Final Equation
# -----------------------------------------

print("\n==============================")
print("FINAL MODEL")
print("==============================")

print("Slope (m):", m)
print("Constant (c):", c)

print("Final Equation:")
print("y =", m, "x +", c)


# -----------------------------------------
# 9. Prediction
# -----------------------------------------

# Suppose a student studied for 7 hours
hours = 7

# Apply y = mx + c
prediction = m * hours + c

print("\nStudent studied:", hours, "hours")
print("Predicted Marks:", prediction)
