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


def compute_error(h, y, theta, landa):
    l2norm = landa * np.sum(np.power(theta, 2))
    return (-y * np.log(h) - (1 - y) * np.log(1 - h) + l2norm).mean()


def logistic_regression(init_x, init_y, learning_rate, num_of_iters, landa):
    theta = np.zeros((init_x.shape[1], 1))
    for i in range(num_of_iters):
        z = np.dot(init_x, theta)
        h = sigmoid(z)
        l2norm = landa * theta
        gradient = np.dot(init_x.T, (h - init_y.T)) + l2norm
        theta -= learning_rate / init_y.size * gradient
        if i % 10000 == 0:
            print('compute_error: %f' % compute_error(h, init_y, theta, landa))
    return theta


def predict(data_in, coeffs, threshold):
    return sigmoid(np.dot(data_in, coeffs)) >= threshold


def main():
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
    num_of_iters = 100000
    learning_rate = 0.001
    landa = 0.0
    T = logistic_regression(X, Y, learning_rate, num_of_iters, landa)
    print(T)
    final_z = np.dot(X, T)
    final_h = sigmoid(final_z)
    print('Final Error: %f' % compute_error(final_h, Y, T, landa))


if __name__ == "__main__":
    main()
