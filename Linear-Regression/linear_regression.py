import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def fit_line(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    nume = np.sum((x - x_mean) * (y - y_mean))
    deno = np.sum((x - x_mean) ** 2)
    w = nume / deno
    b = y_mean - w * x_mean
    return w, b


x = np.array([1, 3, 8, 6, 2])
y_clean = 3 * x + 2

w_clean, b_clean = fit_line(x, y_clean)
print("clean fit: w =", w_clean, " b =", b_clean)
print("true equation was y = 3x + 2, fit gave y =", w_clean, "x +", b_clean)

noise = np.random.uniform(0, 1, len(x))
y_noisy = y_clean + noise

w_noisy, b_noisy = fit_line(x, y_noisy)
print()
print("noisy fit: w =", w_noisy, " b =", b_noisy)
print("true equation was y = 3x + 2, fit gave y =", w_noisy, "x +", b_noisy)
print("difference from true w:", abs(w_noisy - 3), " difference from true b:", abs(b_noisy - 2))


iris = load_iris()
X = iris.data
y = iris.target

X_class1 = X[y == 0]
X_class2 = X[(y == 1) | (y == 2)]

X1_train, X1_test = train_test_split(X_class1, test_size=10, train_size=40, random_state=42)
X2_train, X2_test = train_test_split(X_class2, test_size=20, train_size=80, random_state=42)

print()
print("class I train/test:", X1_train.shape[0], X1_test.shape[0])
print("class II train/test:", X2_train.shape[0], X2_test.shape[0])

X_train = np.vstack([X1_train, X2_train])
y_train = np.concatenate([np.ones(len(X1_train)), -np.ones(len(X2_train))])

X_test = np.vstack([X1_test, X2_test])
y_test = np.concatenate([np.ones(len(X1_test)), -np.ones(len(X2_test))])

X_train_b = np.c_[np.ones(X_train.shape[0]), X_train]
w = np.linalg.pinv(X_train_b).dot(y_train)

X_test_b = np.c_[np.ones(X_test.shape[0]), X_test]
raw_pred = X_test_b.dot(w)
y_pred = np.where(raw_pred >= 0, 1, -1)

print()
print("class I vs class II accuracy:", accuracy_score(y_test, y_pred))


def train_one_vs_all(X, y, target_class):
    y_ova = np.where(y == target_class, 1, -1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_ova, test_size=0.2, random_state=42)
    X_tr_b = np.c_[np.ones(X_tr.shape[0]), X_tr]
    w = np.linalg.pinv(X_tr_b).dot(y_tr)
    X_te_b = np.c_[np.ones(X_te.shape[0]), X_te]
    pred = np.where(X_te_b.dot(w) >= 0, 1, -1)
    acc = accuracy_score(y_te, pred)
    return w, acc


w1, acc1 = train_one_vs_all(X, y, 1)
w2, acc2 = train_one_vs_all(X, y, 2)

print()
print("one vs all, class 1 vs rest accuracy:", acc1)
print("one vs all, class 2 vs rest accuracy:", acc2)
