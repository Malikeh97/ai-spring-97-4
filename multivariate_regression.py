import numpy as np
import matplotlib.pyplot as plt

COLNUM = (0, 2, 13)


# def univariate_regression():

def read_file():
    with open('housing.data') as f:
        content = f.readlines()
    content = [x.strip().split() for x in content]
    for i in range(len(content)):
        content[i] = [float(x) for x in content[i]]
    return content


def compute_error(init_B, init_x, init_y):
    tmp = init_y - (init_B.T * init_x.T)
    totalError = np.sum(np.power(tmp, 2))
    return totalError / (float(len(init_y)) * 2)


def step_gradient(init_B, init_x, init_y, lrate):
    N = float(init_y.shape[1])
    tmp = init_y - (np.dot(init_B.T, init_x.T))
    B_gradient = -(1 / N) * (np.dot(tmp, init_x))
    new_B = init_B - (lrate * B_gradient.T)
    return new_B


def multivariate_regression(init_x, init_y, lrate, iters):
    # fig1 = plt.figure(1)
    # ax = fig1.add_subplot(111)
    # plt.axis([0, x.max(), 0, given_y.max()])
    # plt.xlabel('CRIM')
    # plt.ylabel('MEDV')
    # x_arr = np.array(x)[0]
    # y_arr = np.array(given_y)[0]
    # _, line = ax.plot(x_arr, y_arr, 'bo', x_arr, m * x_arr + b, 'r')

    new_B = np.ones((14, 1))
    errors = [compute_error(new_B, init_x, init_y)]
    for i in range(iters):
        # print("---")
        # print(new_B)
        new_B = step_gradient(new_B, init_x, init_y, lrate)
        errors.append(compute_error(new_B, init_x, init_y))
        # if i % 50 == 0:
        #     line.set_ydata(m * x_arr + b)
        #     fig1.canvas.draw()
        #     plt.pause(0.01)
        #     print(i)

    return new_B, errors


if __name__ == "__main__":
    inputs = read_file()
    X = np.ones((506, 1))
    T = np.matrix([x[0:13] for x in inputs])
    T = (T - np.mean(T, axis=0)) / np.std(T, axis=0)  # normalized
    X = np.append(X, T, axis=1)
    Y = np.matrix([x[13] for x in inputs])
    Y = (Y - np.mean(Y, axis=1)) / np.std(Y, axis=1)  # normalized
    learning_rate = 0.001
    num_of_iter = 10000
    # plt.plot(x2, y, 'bo')
    # plt.axis([0, x1.max(), 0, y.max()])
    # plt.ylabel('MEDV')
    # plt.xlabel('CRIM')
    # plt.show()
    B, errors = multivariate_regression(X, Y, learning_rate, num_of_iter)

    fig1 = plt.figure(1)
    y_predict = np.dot(B.T, X.T)
    plt.axis([0, Y.max(), 0, y_predict.max()])
    plt.xlabel('CRIM')
    plt.ylabel('MEDV')
    y_arr = np.array(Y)[0]
    y_predict_arr = np.array(y_predict)[0]
    plt.plot(y_arr, y_predict_arr, 'ro')

    plt.figure(2)
    plt.axis([0, num_of_iter, 0, errors[0]])
    plt.xlabel('iter')
    plt.ylabel('err')
    plt.plot(errors)
    plt.show()
    sum = 0
    # plt.pause(10000)
    print(B)
