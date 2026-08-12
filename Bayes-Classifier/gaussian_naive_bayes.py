import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("train samples", X_train.shape[0])
print("test samples", X_test.shape[0])
print("train dist", np.bincount(y_train))
print("test dist", np.bincount(y_test))

gnb = GaussianNB().fit(X_train, y_train)

print("training done")
for cls, prior in zip(iris.target_names, gnb.class_prior_):
    print(cls, round(prior, 4))

y_pred = gnb.predict(X_test)
y_prob = gnb.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)

print()
print("test accuracy", acc * 100)
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cv_scores = cross_val_score(GaussianNB(), X, y, cv=5)
print("cv scores", (cv_scores * 100).round(2))
print("mean", cv_scores.mean() * 100, "std", cv_scores.std() * 100)


def gaussian_pdf(x, mu, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * (x - mu) ** 2 / var)


x_sample = X_test[0]

scores = []
for k in range(3):
    likelihood = np.prod([gaussian_pdf(x_sample[j], gnb.theta_[k, j], gnb.var_[k, j]) for j in range(4)])
    scores.append(likelihood * gnb.class_prior_[k])

pred_manual = iris.target_names[np.argmax(scores)]
pred_lib = iris.target_names[gnb.predict([x_sample])[0]]

print()
print("manual calc gives", pred_manual)
print("library gives", pred_lib)
print("same answer?", pred_manual == pred_lib)
