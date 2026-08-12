import numpy as np
from sklearn.datasets import load_iris, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier

kernels = ['linear', 'poly', 'rbf']

iris = load_iris()
x = iris.data
y = iris.target

print("iris shape", x.shape)
print("classes", iris.target_names)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

print("train", x_train.shape, "test", x_test.shape)

iris_results = {}
for k in kernels:
    model = SVC(kernel=k, random_state=42)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)
    iris_results[k] = acc
    print(k, "->", acc)

best_kernel = max(iris_results, key=iris_results.get)
model_best = SVC(kernel=best_kernel, random_state=42)
model_best.fit(x_train, y_train)
pred_best = model_best.predict(x_test)

print()
print("best kernel on iris was", best_kernel)
print(classification_report(y_test, pred_best, target_names=iris.target_names))


# now the mnist part, one-vs-rest since its 10 classes not 2
# this part needs internet the first time, it downloads the dataset and that can take a bit

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
x_mnist = mnist.data
y_mnist = mnist.target.astype(int)

print("mnist full size", x_mnist.shape)

# using the full 70k would take forever to train an SVM on, so just grabbing a subset
x_mnist = x_mnist[:10000]
y_mnist = y_mnist[:10000]

scaler_m = StandardScaler()
x_mnist_scaled = scaler_m.fit_transform(x_mnist)

x_train_m, x_test_m, y_train_m, y_test_m = train_test_split(
    x_mnist_scaled, y_mnist, test_size=0.2, random_state=42
)

print("mnist subset train", x_train_m.shape, "test", x_test_m.shape)

mnist_results = {}

ova_linear = OneVsRestClassifier(SVC(kernel='linear', random_state=42))
ova_linear.fit(x_train_m, y_train_m)
pred_linear = ova_linear.predict(x_test_m)
mnist_results['linear'] = accuracy_score(y_test_m, pred_linear)
print("linear kernel done, acc", mnist_results['linear'])

ova_poly = OneVsRestClassifier(SVC(kernel='poly', degree=3, random_state=42))
ova_poly.fit(x_train_m, y_train_m)
pred_poly = ova_poly.predict(x_test_m)
mnist_results['poly'] = accuracy_score(y_test_m, pred_poly)
print("poly kernel done, acc", mnist_results['poly'])

ova_rbf = OneVsRestClassifier(SVC(kernel='rbf', random_state=42))
ova_rbf.fit(x_train_m, y_train_m)
pred_rbf = ova_rbf.predict(x_test_m)
mnist_results['rbf'] = accuracy_score(y_test_m, pred_rbf)
print("rbf kernel done, acc", mnist_results['rbf'])

preds_by_kernel = {'linear': pred_linear, 'poly': pred_poly, 'rbf': pred_rbf}

print()
print("kernel comparison")
for k in kernels:
    print(k, "iris:", iris_results[k], " mnist:", mnist_results[k])

best_mnist = max(mnist_results, key=mnist_results.get)
print()
print("best kernel on mnist was", best_mnist)
print(classification_report(y_test_m, preds_by_kernel[best_mnist]))
