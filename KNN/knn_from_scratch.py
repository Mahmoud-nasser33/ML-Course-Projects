import numpy as np
import sklearn.datasets as datasets
import sklearn.model_selection as model_selection


class KNN:
    def __init__(self, x, y, k, classification=True):
        self.x_train = x
        self.y_train = y
        self.k = k
        self.classification = classification

    def euclidean_distance(self, inst1, inst2):
        return np.sqrt(np.sum((inst1 - inst2) ** 2))

    def _get_k_neighbors(self, inst1):
        distances = [self.euclidean_distance(inst1, inst2) for inst2 in self.x_train]
        distances = np.asarray(distances)
        indices = np.argpartition(distances, self.k)
        return indices[:self.k]

    def predict_class(self, inst1):
        knn_indices = self._get_k_neighbors(inst1)
        knn_labels = [self.y_train[i] for i in knn_indices]
        if self.classification:
            occurrences = np.bincount(knn_labels)
            return np.argmax(occurrences)
        else:
            return np.mean(knn_labels)

    def get_accuracy(self, y_test, predictions):
        y_test = np.array(y_test)
        predictions = np.array(predictions)
        correct = np.sum(y_test == predictions)
        return (correct / len(y_test)) * 100.0


def evaluate_knn(x_train, x_test, y_train, y_test, k_values, label=""):
    results = {}
    for k in k_values:
        clf = KNN(x_train, y_train, k)
        preds = [clf.predict_class(inst) for inst in x_test]
        acc = clf.get_accuracy(y_test, preds)
        results[k] = acc
        print(label, "k =", k, "accuracy =", acc)
    return results


iris = datasets.load_iris()
x = iris.data
y = iris.target

print("dataset loaded", x.shape[0], "samples", x.shape[1], "features")

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.20, shuffle=True, random_state=42
)

print("train samples", len(x_train), "test samples", len(x_test))

k_values = [3, 5, 7]

print()
print("without normalization")
raw_results = evaluate_knn(x_train, x_test, y_train, y_test, k_values, label="[raw]")

x_min = x_train.min(axis=0)
x_max = x_train.max(axis=0)
x_train_norm = (x_train - x_min) / (x_max - x_min)
x_test_norm = (x_test - x_min) / (x_max - x_min)

print()
print("with min-max normalization")
norm_results = evaluate_knn(x_train_norm, x_test_norm, y_train, y_test, k_values, label="[norm]")

print()
print("comparison")
for k in k_values:
    delta = norm_results[k] - raw_results[k]
    print("k =", k, "raw =", raw_results[k], "norm =", norm_results[k], "delta =", delta)
