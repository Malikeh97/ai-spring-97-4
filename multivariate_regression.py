import numpy as np
import matplotlib.pyplot as plt


def read_file():
    with open('housing.data') as f:
        content = f.readlines()
    content = [x.strip().split() for x in content]
    for i in range(len(content)):
        content[i] = [float(x) for x in content[i]]
    return content


def mean_average_error(init_B, init_x, init_y):
    n = float(init_y.shape[1])
    e = init_y - (init_B.T * init_x.T)
    total_error = np.sum(np.abs(e))
    return total_error / n


def compute_error(init_B, init_x, init_y, landa):
    n = float(init_y.shape[1])
    tmp = init_y - (init_B.T * init_x.T)
    total_error = np.sum(np.power(tmp, 2))
    l2norm = landa / 2 * np.sum(np.power(init_B, 2))
    return total_error / (n * 2) + l2norm


def step_gradient(init_B, init_x, init_y, learning_rate, landa):
    n = float(init_y.shape[1])
    tmp = init_y - (np.dot(init_B.T, init_x.T))
    l2norm = landa * init_B
    B_gradient = -(1 / n) * np.dot(tmp, init_x).T + l2norm
    new_B = init_B - (learning_rate * B_gradient)
    return new_B


def multivariate_regression(init_x, init_y, learning_rate, num_of_iter, landa=None):
    new_B = np.ones((14, 1))
    errors = [compute_error(new_B, init_x, init_y, landa)]
    for i in range(num_of_iter):
        new_B = step_gradient(new_B, init_x, init_y, learning_rate, landa)
        errors.append(compute_error(new_B, init_x, init_y, landa))

    return new_B, errors


def main():
    inputs = read_file()
    X = np.ones((506, 1))
    T = np.matrix([x[0:13] for x in inputs])
    T = (T - np.mean(T, axis=0)) / np.std(T, axis=0)  # normalized
    X = np.append(X, T, axis=1)
    Y = np.matrix([x[13] for x in inputs])
    Y = (Y - np.mean(Y, axis=1)) / np.std(Y, axis=1)  # normalized

    num_of_iter = 2000
    learning_rate = 0.01
    landa = 0.01
    B, errors = multivariate_regression(X, Y, learning_rate, num_of_iter, landa)
    print(B.T)
    print('MSE = %.20f' % np.asscalar(errors[-1]))
    print('MAE = %.20f' % mean_average_error(B, X, Y))

    plt.figure(1)
    plt.subplot(211)
    y_predict = np.dot(B.T, X.T)
    plt.axis([Y.min() - 0.25, Y.max() + 0.25, y_predict.min() - 0.25, y_predict.max() + 0.25])
    y_arr = np.array(Y)[0]
    y_predict_arr = np.array(y_predict)[0]
    plt.plot(y_arr, y_predict_arr, 'ro')

    plt.subplot(212)
    plt.axis([0, num_of_iter, 0, errors[0]])
    plt.xlabel('iter')
    plt.ylabel('err')
    plt.plot(errors)
    plt.show()


if __name__ == "__main__":
    main()
