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


def compute_error_for_line_given_points(b, m, x, y):
    totalError = np.sum(np.power(y - (m * x + b),2))
    return totalError / ( float(len(y)) * 2 )


def step_gradient(b, m, x, given_y, lrate):
    N = float(given_y.shape[1])
    tmp = given_y - ((m * x) + b)
    # tmp = np.power(tmp, 2)
    b_gradient = -(1 / N) * np.sum(tmp)
    m_gradient = -(1 / N) * tmp * x.T
    current_b = b - (lrate * b_gradient)
    current_m = m - (lrate * m_gradient[0,0])
    return current_b, current_m


def univariate_regression(x, given_y, lrate, iters):
    m = b = 0

    plt.ion()
    fig1 = plt.figure(1)
    ax = fig1.add_subplot(111)
    plt.axis([0, x.max(), 0, given_y.max()])
    plt.xlabel('CRIM')
    plt.ylabel('MEDV')
    x_arr = np.array(x)[0]
    y_arr = np.array(given_y)[0]
    _, line = ax.plot(x_arr, y_arr, 'bo', x_arr, m * x_arr + b, 'r')

    errors = [compute_error_for_line_given_points(b, m, x, given_y)]
    for i in range(iters):
        b, m = step_gradient(b, m, x, given_y, lrate)

        if i % 50 == 0:
            line.set_ydata(m * x_arr + b)
            fig1.canvas.draw()
            plt.pause(0.01)
            print(i)

        errors.append(compute_error_for_line_given_points(b, m, x, given_y))
    return b, m, errors


if __name__ == "__main__":
    inputs = read_file()
    x1 = np.matrix([x[COLNUM[0]] for x in inputs])
    x2 = np.matrix([x[COLNUM[1]] for x in inputs])
    y = np.matrix([x[COLNUM[2]] for x in inputs])
    learning_rate = 0.001
    num_of_iter = 7000
    # plt.plot(x2, y, 'bo')
    # plt.axis([0, x1.max(), 0, y.max()])
    # plt.ylabel('MEDV')
    # plt.xlabel('CRIM')
    # plt.show()
    b1, m1, errors1 = univariate_regression(x1, y, learning_rate, num_of_iter)
    # b, m = univariate_regression(x2, y, b, m, learning_rate, num_of_iter)

    plt.figure(2)
    plt.axis([0, num_of_iter, 0, errors1[0]])
    plt.xlabel('iter')
    plt.ylabel('err')
    plt.plot(errors1)
    plt.show()
    plt.pause(10000)
    err = compute_error_for_line_given_points(b, m, x1, y)
    print(err)
