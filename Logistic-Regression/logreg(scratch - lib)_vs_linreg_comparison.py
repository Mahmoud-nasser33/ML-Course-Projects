import numpy as np
from sklearn.datasets import load_iris, fetch_openml
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

np.random.seed(42)


class LogisticRegressionScratch:

    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.W = None
        self.losses = []

    def softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / e_z.sum(axis=1, keepdims=True)

    def cross_entropy(self, y_hat, y_ohe):
        eps = 0.000000001
        return -np.mean(np.sum(y_ohe * np.log(y_hat + eps), axis=1))

    def add_bias(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def one_hot(self, y, n_classes):
        ohe = np.zeros((len(y), n_classes))
        ohe[np.arange(len(y)), y] = 1
        return ohe

    def fit(self, X, y):
        X = self.add_bias(X)
        N, n_features = X.shape
        n_classes = len(np.unique(y))

        y_ohe = self.one_hot(y, n_classes)
        self.W = np.zeros((n_features, n_classes))

        for _ in range(self.epochs):
            z = X @ self.W
            y_hat = self.softmax(z)
            grad = (1 / N) * X.T @ (y_hat - y_ohe)
            self.W -= self.lr * grad
            self.losses.append(self.cross_entropy(y_hat, y_ohe))

    def predict(self, X):
        X = self.add_bias(X)
        return np.argmax(self.softmax(X @ self.W), axis=1)


iris = load_iris()
X_iris = iris.data
y_iris = iris.target

scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

X_tr, X_te, y_tr, y_te = train_test_split(X_iris_scaled, y_iris, test_size=0.2, random_state=42)

lr_scratch_iris = LogisticRegressionScratch(lr=0.1, epochs=1000)
lr_scratch_iris.fit(X_tr, y_tr)
pred_scratch_iris = lr_scratch_iris.predict(X_te)
acc_scratch_iris = accuracy_score(y_te, pred_scratch_iris)
print("scratch logreg on iris:", acc_scratch_iris)

lr_lib_iris = LogisticRegression(max_iter=1000, random_state=42)
lr_lib_iris.fit(X_tr, y_tr)
pred_lib_iris = lr_lib_iris.predict(X_te)
acc_lib_iris = accuracy_score(y_te, pred_lib_iris)
print("sklearn logreg on iris:", acc_lib_iris)

lin_iris = LinearRegression()
lin_iris.fit(X_tr, y_tr)
pred_lin_iris = np.round(lin_iris.predict(X_te)).clip(0, 2).astype(int)
acc_lin_iris = accuracy_score(y_te, pred_lin_iris)
print("linear reg (rounded) on iris:", acc_lin_iris)

print()
print("iris summary")
print("scratch  ", acc_scratch_iris)
print("sklearn  ", acc_lib_iris)
print("linreg   ", acc_lin_iris)


mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_mnist = mnist.data[:10000].astype(float)
y_mnist = mnist.target[:10000].astype(int)

X_mnist /= 255.0

X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(X_mnist, y_mnist, test_size=0.2, random_state=42)

lr_scratch_mnist = LogisticRegressionScratch(lr=0.1, epochs=300)
lr_scratch_mnist.fit(X_tr_m, y_tr_m)
pred_scratch_mnist = lr_scratch_mnist.predict(X_te_m)
acc_scratch_mnist = accuracy_score(y_te_m, pred_scratch_mnist)
print("\nscratch logreg on mnist:", acc_scratch_mnist)

lr_lib_mnist = LogisticRegression(max_iter=300, solver='saga', random_state=42)
lr_lib_mnist.fit(X_tr_m, y_tr_m)
pred_lib_mnist = lr_lib_mnist.predict(X_te_m)
acc_lib_mnist = accuracy_score(y_te_m, pred_lib_mnist)
print("sklearn logreg on mnist:", acc_lib_mnist)

lin_mnist = LinearRegression()
lin_mnist.fit(X_tr_m, y_tr_m)
pred_lin_mnist = np.round(lin_mnist.predict(X_te_m)).clip(0, 9).astype(int)
acc_lin_mnist = accuracy_score(y_te_m, pred_lin_mnist)
print("linear reg (rounded) on mnist:", acc_lin_mnist)

print()
print("mnist summary")
print("scratch  ", acc_scratch_mnist)
print("sklearn  ", acc_lib_mnist)
print("linreg   ", acc_lin_mnist)

print()
print("final comparison, iris vs mnist")
print("scratch logreg", acc_scratch_iris, acc_scratch_mnist)
print("sklearn logreg", acc_lib_iris, acc_lib_mnist)
print("linear regression", acc_lin_iris, acc_lin_mnist)
