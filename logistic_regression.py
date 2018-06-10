import numpy as np
import matplotlib.pyplot as plt


def read_file():
    with open('logistic_and_svm_data.txt') as f:
        content = f.readlines()
    content = [x.strip().split(',') for x in content]
    for i in range(len(content)):
        content[i] = [float(x) for x in content[i]]
    return content


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_error(h, y, theta, landa=None):
    l2norm = landa / 2 * np.sum(np.power(theta, 2)) if landa is not None else 0
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()+ l2norm


def logistic_regression(init_x, init_y, landa=None):
    theta = np.zeros((init_x.shape[1], 1))
    for i in range(num_iter):
        z = np.dot(init_x, theta)
        h = sigmoid(z)
        l2norm = landa * theta if landa is not None else 0
        gradient = np.dot(init_x.T, (h - init_y.T)) / init_y.size + l2norm
        theta -= lr * gradient
        if i % 10000 == 0:
            print('compute_error: %f' % compute_error(h, init_y, theta, landa))
    return theta

def predict(data_in, coeffs, threshold):
    return sigmoid(np.dot(data_in, coeffs)) >= threshold

if __name__ == "__main__":
    inputs = read_file()
    zeros_x = [x[0] for x in inputs if x[2] == 0]
    zeros_y = [x[1] for x in inputs if x[2] == 0]
    ones_x = [x[0] for x in inputs if x[2] == 1]
    ones_y = [x[1] for x in inputs if x[2] == 1]

    X = np.matrix([x[0:2] for x in inputs])
    intercept = np.ones((X.shape[0], 1))
    X = np.concatenate((intercept, X), axis=1)
    Y = np.matrix([x[2] for x in inputs])

    plt.figure(1)
    plt.axis([np.matrix(inputs)[:, 0].min() - 5, np.matrix(inputs)[:, 0].max() + 5, np.matrix(inputs)[:, 1].min() - 5,
              np.matrix(inputs)[:, 1].max() + 5])
    plt.plot(zeros_x, zeros_y, 'bo', ones_x, ones_y, 'r*')
    plt.show()
    landa = 0.01
    lr = 0.001
    num_iter = 100000
    T = logistic_regression(X, Y, landa)
    print(T)
    final_z = np.dot(X, T)
    final_h = sigmoid(final_z)
    print('Final Error: %f'% compute_error(final_h, Y, T, landa))
