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


def mean_average_error(b, m, x, y):
    n = float(y.shape[1])
    tmp = np.abs(y - (m * x + b))
    total_error = np.sum(tmp)
    return total_error / n


def compute_error_for_line_given_points(b, m, x, y):
    n = float(y.shape[1])
    total_error = np.sum(np.power(y - (m * x + b), 2))
    return total_error / (n * 2)


def step_gradient(b, m, x, y, learning_rate):
    n = float(y.shape[1])
    tmp = y - ((m * x) + b)
    # tmp = np.power(tmp, 2)
    b_gradient = -(1 / n) * np.sum(tmp)
    m_gradient = -(1 / n) * tmp * x.T
    current_b = b - (learning_rate * b_gradient)
    current_m = m - (learning_rate * m_gradient[0, 0])
    return current_b, current_m


def univariate_regression(x, y, learning_rate, num_of_iter):
    m = b = 0

    errors = [compute_error_for_line_given_points(b, m, x, y)]
    for i in range(num_of_iter):
        b, m = step_gradient(b, m, x, y, learning_rate)
        errors.append(compute_error_for_line_given_points(b, m, x, y))
    return b, m, errors


def main():
    inputs = read_file()
    x1 = np.matrix([x[COLNUM[0]] for x in inputs])
    x2 = np.matrix([x[COLNUM[1]] for x in inputs])
    y = np.matrix([x[COLNUM[2]] for x in inputs])
    learning_rate = 0.01
    num_of_iter = 10000

    # plt.figure(1)
    # plt.plot(x1, y, 'bo')
    # plt.axis([-5, x1.max() + 5, -5, y.max() + 5])
    # plt.ylabel('Price')
    # plt.xlabel('Crime')
    #
    # plt.figure(2)
    # plt.plot(x2, y, 'bo')
    # plt.axis([-5, x2.max() + 5, -5, y.max() + 5])
    # plt.ylabel('Price')
    # plt.xlabel('Tax')
    # plt.show()

    b1, m1, errors1 = univariate_regression(x1, y, learning_rate, num_of_iter)
    b2, m2, errors2 = univariate_regression(x2, y, learning_rate, num_of_iter)

    print('-------- 1 --------')
    print('y = %fx + %f' % (m1, b1))
    print('MSE = %f' % np.asscalar(errors1[-1]))
    print('MAE = %f' % mean_average_error(b1, m1, x1, y))

    print('-------- 2 --------')
    print('y = %fx + %f' % (m2, b2))
    print('MSE = %f' % np.asscalar(errors2[-1]))
    print('MAE = %f' % mean_average_error(b2, m2, x2, y))

    plt.figure(1)
    plt.title('Crime')
    plt.subplot(211)
    plt.axis([0, x1.max(), 0, y.max()])
    x_arr = np.array(x1)[0]
    y_arr = np.array(y)[0]
    plt.plot(x_arr, y_arr, 'bo', x_arr, m1 * x_arr + b1, 'r')

    plt.subplot(212)
    plt.axis([0, num_of_iter, 0, errors1[0]])
    plt.xlabel('iter')
    plt.ylabel('err')
    plt.plot(errors1)

    plt.figure(2)
    plt.title('Tax')
    plt.subplot(211)
    plt.axis([0, x2.max(), 0, y.max()])
    x_arr = np.array(x2)[0]
    y_arr = np.array(y)[0]
    plt.plot(x_arr, y_arr, 'bo', x_arr, m2 * x_arr + b2, 'r')

    plt.subplot(212)
    plt.axis([0, num_of_iter, 0, errors2[0]])
    plt.xlabel('iter')
    plt.ylabel('err')
    plt.plot(errors2)
    plt.show()


if __name__ == "__main__":
    main()
