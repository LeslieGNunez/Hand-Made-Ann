import numpy as np

# -----------------------------
# Simple 2-Layer ANN with NumPy
# -----------------------------

class SimpleANN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        np.random.seed(42)

        # Weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

        self.learning_rate = learning_rate

    # Activation function: sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative of sigmoid
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Feedforward
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_hat = self.sigmoid(self.z2)

        return self.y_hat

    # Loss function: Mean Squared Error
    def compute_loss(self, y, y_hat):
        return np.mean((y - y_hat) ** 2)

    # Backpropagation
    def backward(self, X, y):
        m = X.shape[0]

        # Output layer error
        output_error = self.y_hat - y
        output_delta = output_error * self.sigmoid_derivative(self.y_hat)

        dW2 = np.dot(self.a1.T, output_delta) / m
        db2 = np.sum(output_delta, axis=0, keepdims=True) / m

        # Hidden layer error
        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)

        dW1 = np.dot(X.T, hidden_delta) / m
        db1 = np.sum(hidden_delta, axis=0, keepdims=True) / m

        # Gradient descent update
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    # Train network
    def train(self, X, y, epochs=5000):
        for epoch in range(epochs):
            y_hat = self.forward(X)
            loss = self.compute_loss(y, y_hat)
            self.backward(X, y)

            if epoch % 500 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    # Predict
    def predict(self, X):
        return self.forward(X)


# -----------------------------------------
# Example: Predict next number in a sequence
# -----------------------------------------
# Using 3 numbers to predict the 4th number

X = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8]
], dtype=float)

y = np.array([
    [4],
    [5],
    [6],
    [7],
    [8],
    [9]
], dtype=float)

# Normalize data for better sigmoid performance
X = X / 10.0
y = y / 10.0

# Create and train ANN
ann = SimpleANN(input_size=3, hidden_size=5, output_size=1, learning_rate=0.5)
ann.train(X, y, epochs=5000)

# User input
print("\nEnter 3 numbers in a sequence to predict the next one.")
user_input = input("Example: 7 8 9\nInput: ")

numbers = list(map(float, user_input.split()))

if len(numbers) != 3:
    print("Please enter exactly 3 numbers.")
else:
    user_X = np.array([numbers]) / 10.0
    prediction = ann.predict(user_X)
    predicted_value = prediction[0][0] * 10.0

    print(f"Predicted next number: {predicted_value:.2f}")
