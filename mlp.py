import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def circle_points(num_points=100, lower_bound=-3, upper_bound = 3, radius = 2):
  np.random.seed(42)

  X = np.random.uniform(lower_bound, upper_bound, (num_points, 2))
  y = (np.sqrt(X[:, 0]**2 + X[:, 1]**2) < radius).astype(int).reshape(-1, 1)

  return X, y

def print_circle(X, y):
  plt.scatter(X[:,0], X[:,1], c=y[:,0], cmap='bwr', alpha=0.6)
  plt.title("RED - 1, BLUE - 0")
  plt.xlabel("x")
  plt.ylabel("y")
  plt.show()


def split_test_train(X, y, test_size=0.2):
  split_index = int(len(X) * (1 - test_size))

  X_train = X[:split_index]
  X_test = X[split_index:]
  y_train = y[:split_index]
  y_test = y[split_index:]

  return X_train, X_test, y_train, y_test

def plot_test_predictions(X_test, model, circle_radius=2):
    predictions = []
    for x in X_test:
        pred = model.predict(x.reshape(-1, 1))
        predictions.append(pred.item())

    predictions = np.array(predictions)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_test[:, 0], X_test[:, 1],
        c=predictions,
        cmap='bwr', 
        vmin=0, vmax=1,
        s=60,
        edgecolors='k'
    )

    
    circle = patches.Circle((0, 0), circle_radius, edgecolor='green', facecolor='none', linewidth=2, linestyle='--')
    plt.gca().add_patch(circle)

    plt.colorbar(scatter, label='Prediction (0=Blue, 1=Red)')
    plt.title("Model Predictions on Test Data with True Circle")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis('equal') 
    plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

#MSE
def loss(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def d_loss(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)



class MLP:

    def __init__(self, hidden_layer_size=2, num_of_hidden_layers=2, input_size=2, output_size=1):

        self.weights = []
        self.biases = []

        layer_sizes = [input_size] + [hidden_layer_size] * num_of_hidden_layers + [output_size]

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * 0.1  
            b = np.zeros((layer_sizes[i + 1], 1)) 
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):

        self.activations = [x]  # Activations: start with input

        a = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b  
            a = sigmoid(z)

            self.activations.append(a)

        return a



    def backward(self, y_true):
    
        error = d_loss(y_true, self.activations[-1])
        delta = error * d_sigmoid(self.activations[-1])  

        deltas = [delta]  

        for i in reversed(range(len(self.weights) - 1)):  
            sp = d_sigmoid(self.activations[i + 1]) 
            delta = np.dot(self.weights[i + 1].T, deltas[-1]) * sp  
            deltas.append(delta) 

        deltas.reverse() 

        return deltas  

    def update_parameters(self, deltas, lr):

        for i in range(len(self.weights)):
            dw = np.dot(deltas[i], self.activations[i].T)  # Weight gradient: dw = delta * activation^T
            db = np.sum(deltas[i], axis=1, keepdims=True)  # Bias gradient: db = delta

            self.weights[i] -= lr * dw  # W = W - lr * dW
            self.biases[i] -= lr * db  # b = b - lr * db

    def train(self, X, Y, epochs, lr):

        for epoch in range(epochs):
            losses =[]
            for x, y in zip(X, Y):

                x = x.reshape(-1, 1)
                y_true = y.reshape(-1, 1)

                y_pred = self.forward(x)

                loss_ = loss(y_true, y_pred)
                losses.append(loss_)

                deltas = self.backward(y_true)

                self.update_parameters(deltas, lr)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs},  Avg Loss: {np.mean(losses)}")


    def predict(self, x):
        return self.forward(x)

def main():

    X, y = circle_points()
    print_circle(X, y)

    X_train, X_test, y_train, y_test = split_test_train(X, y)


    mlp = MLP(num_of_hidden_layers=1, hidden_layer_size=6, input_size=2, output_size=1)
    mlp.train(X_train, y_train, 1000, 0.05)

    for i in range(len(X_test)):
        print(f"Input: {X_test[i]}, Prediction: {mlp.predict(X_test[i].reshape(-1, 1))}, True: {y_test[i]}")

    plot_test_predictions(X_test, mlp)



main()