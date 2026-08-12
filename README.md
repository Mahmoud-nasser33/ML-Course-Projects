# ML Course Projects

This is a dump of the assignments I did for my ML course this semester. Six
folders, mostly built around the Iris dataset (one of them also uses
MNIST). Each folder has its own script and runs on its own, nothing shared
between them.

The general pattern across most of these: implement something from scratch
with plain numpy first, then compare it against the same model from
sklearn. That's usually the actual point of these assignments — seeing
what the library is doing.

## Folders

**KNN/** — two versions of the same comparison, one with KNN written from
scratch (euclidean distance, majority vote) and one using
KNeighborsClassifier. Both tested at k=3, 5, 7, on raw features and then
after min-max normalization.

**perceptron_iris_classification/** — Perceptron built
from scratch (the classic mistake-driven update rule) vs sklearn's
logistic regression, classifying Setosa vs Versicolor.

**linear-Regression/** — First part fits a straight line to
5 points with a known equation, once clean and once with noise added, and
checks how close the fit gets to the real equation. Second part uses that
same least-squares approach as a classifier on Iris, with the exact 40/10
and 80/20 per-class train/test split the assignment asked for, plus an
optional one-vs-all extension.

**SVM/** — SVM with linear, poly, and rbf kernels, on Iris
directly as multi-class and on a chunk of MNIST wrapped in one-vs-all on
purpose (not relying on sklearn doing multi-class automatically, since
that was part of the requirement). MNIST needs internet the first time
you run it since it pulls from OpenML, so I didn't hardcode numbers for that part.

**logistic-regression-scratch-vs-library/** — Logistic regression
from scratch (softmax, one-hot labels, gradient descent on
cross-entropy) compared against sklearn's version, and both compared
against plain linear regression rounded to the nearest class, mostly to
show why that's not really a good idea even though it happens to work on
easy data.

**Bayes-Classifier/** — Gaussian Naive Bayes on Iris,
calculates the Gaussian likelihood by hand for one sample just to check it
actually matches what the library predicts, and it does.
