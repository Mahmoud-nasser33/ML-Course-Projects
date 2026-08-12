import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

np.random.seed(42)


class Perceptron:

    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                net_input = np.dot(xi, self.w) + self.b
                pred = 1 if net_input >= 0 else -1

                if pred != target:
                    update = self.lr * target
                    self.w += update * xi
                    self.b += update
                    errors += 1

            if errors == 0:
                print("converged after", epoch + 1, "epochs")
                break
        else:
            print("didnt fully converge after", self.epochs, "epochs,", errors, "still wrong")

        return self

    def predict(self, X):
        return np.where(np.dot(X, self.w) + self.b >= 0, 1, -1)


def load_binary_iris():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df = df[df["target"] <= 1].copy()
    return df, iris.feature_names


def main():
    df, feature_names = load_binary_iris()
    X = df[feature_names].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("train size", len(X_train))
    print("test size", len(X_test))

    y_train_signed = np.where(y_train == 0, -1, 1)

    perceptron = Perceptron(lr=0.1, epochs=1000)
    perceptron.fit(X_train, y_train_signed)

    perceptron_pred = np.where(perceptron.predict(X_test) == -1, 0, 1)

    print()
    print("PERCEPTRON RESULTS")
    print("acc =", accuracy_score(y_test, perceptron_pred))
    print("recall", recall_score(y_test, perceptron_pred), "  precision", precision_score(y_test, perceptron_pred))
    print("f1:", f1_score(y_test, perceptron_pred))

    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train, y_train)

    log_reg_pred = log_reg.predict(X_test)

    print()
    print("now logistic regression")
    print("accuracy", accuracy_score(y_test, log_reg_pred))
    print("precision:", precision_score(y_test, log_reg_pred))
    print("recall:", recall_score(y_test, log_reg_pred))
    print("f1 score =", f1_score(y_test, log_reg_pred))

    print()
    print("---------------------")
    print("perceptron report")
    print(classification_report(y_test, perceptron_pred, target_names=["Setosa", "Versicolor"]))

    print("log reg report")
    print(classification_report(y_test, log_reg_pred, target_names=["Setosa", "Versicolor"]))


if __name__ == "__main__":
    main()
