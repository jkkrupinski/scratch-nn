import random
import time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def load_mnist():
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data / 255.0, mnist.target.astype(int) 
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_model():
    return MLPClassifier(max_iter=20, random_state=42)

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Raport klasyfikacji:\n", classification_report(y_test, y_pred))

def grid_search(X_train, y_train, X_test, y_test, param_grid):
    start_time = time.time()
    best_score = 0
    best_params = None

    for hidden in param_grid['hidden_layer_sizes']:
        for activation in param_grid['activation']:
            for alpha in param_grid['alpha']:
                model = MLPClassifier(
                    hidden_layer_sizes=hidden,
                    activation=activation,
                    alpha=alpha,
                    max_iter=100,
                    random_state=42
                )
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)

                print(f"Params: hidden={hidden}, activation={activation}, alpha={alpha} -> Accuracy: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_params = (hidden, activation, alpha)

    end_time = time.time()
    elapsed = end_time - start_time
    print("\n Best Grid Search Accuracy:", best_score)
    print("Best Parameters:", best_params)
    print(f" Grid Search Time: {elapsed:.2f} seconds")



def random_search(X_train, y_train, X_test, y_test, param_grid, n_iter=5):
    start_time = time.time()

    best_score = 0
    best_params = None

    for _ in range(n_iter):
        hidden = random.choice(param_grid['hidden_layer_sizes'])
        activation = random.choice(param_grid['activation'])
        alpha = random.choice(param_grid['alpha'])

        model = MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=activation,
            alpha=alpha,
            max_iter=100,
            random_state=42
        )
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        print(f"Params: hidden={hidden}, activation={activation}, alpha={alpha} -> Accuracy: {score:.4f}")

        if score > best_score:
            best_score = score
            best_params = (hidden, activation, alpha)

    end_time = time.time()
    elapsed = end_time - start_time
    print("\n Best Random Search Accuracy:", best_score)
    print("Best Parameters:", best_params)
    print(f"Random Search Time: {elapsed:.2f} seconds")

if __name__ == "__main__":

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01]
    }


    X_train, X_test, y_train, y_test = load_mnist()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = create_model()
    # train_and_evaluate(model, X_train, X_test, y_train, y_test)

    grid_search(X_train, y_train, X_test, y_test, param_grid)
    random_search(X_train, y_train, X_test, y_test, param_grid, n_iter=5)
