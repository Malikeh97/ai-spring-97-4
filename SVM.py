# Mathieu Blondel, October 2010
# License: BSD 3 clause

import numpy as np
import pylab as pl


class SVM(object):

    def __init__(self, T=1, p=3):
        self.T = T
        self.p = p

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # np.hstack((X, np.ones((n_samples, 1))))
        self.alpha = np.zeros(n_samples, dtype=np.float64)

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.polynomial_kernel(X[i], X[j])

        for t in range(self.T):
            for i in range(n_samples):
                if np.sign(np.sum(K[:, i] * self.alpha * y)) != y[i]:
                    self.alpha[i] += 1.0

        # Support vectors
        sv = self.alpha > 1e-5
        ind = np.arange(len(self.alpha))[sv]
        self.alpha = self.alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.alpha),
                                                       n_samples))

    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                s += a * sv_y * self.polynomial_kernel(X[i], sv)
            y_predict[i] = s
        return y_predict

    def predict(self, X):
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        # np.hstack((X, np.ones((n_samples, 1))))
        return np.sign(self.project(X))

    def polynomial_kernel(self, x, y):
        # print(x.shape, y.shape)
        return (1 + np.dot(x, y)) ** self.p


def read_file():
    with open('logistic_and_svm_data.txt') as f:
        content = f.readlines()
    content = [x.strip().split(',') for x in content]
    for i in range(len(content)):
        content[i] = [float(x) for x in content[i]]
    return content


def plot_contour(X1_train, X2_train, svm):  # TODO: it's not a good plot
    pl.plot(X1_train[:, 0], X1_train[:, 1], "ro")
    pl.plot(X2_train[:, 0], X2_train[:, 1], "bo")
    pl.scatter(svm.sv[:, 0], svm.sv[:, 1], s=100, c="g")

    X1, X2 = np.meshgrid(np.linspace(-6, 6, 50), np.linspace(-6, 6, 50))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = svm.project(X).reshape(X1.shape)
    pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')

    pl.axis("tight")
    pl.show()


def main():
    content = read_file()
    train = content[:90]
    test = content[90:]

    # train = sorted(train, key=lambda x: x[2]) # sort training set

    X_train = np.array([x[:2] for x in train])
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    y_train = np.array([x[2] if x[2] == 1 else -1 for x in train])
    X_test = np.array([x[:2] for x in test])
    X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)
    y_test = np.array([x[2] if x[2] == 1 else -1 for x in test])

    # svm = SVM(T=200)  # if training set sorted default p is better

    svm = SVM(T=2000, p=7)
    svm.fit(X_train, y_train)

    y_predict = svm.predict(X_test)
    correct = np.asscalar(np.sum(y_predict == y_test))
    print("%d out of %d predictions correct" % (correct, len(y_predict)))

    plot_contour(X_train[y_train == 1], X_train[y_train == -1], svm)


if __name__ == "__main__":
    main()
