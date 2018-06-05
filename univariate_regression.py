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


def compute_error_for_line_given_points(given_b, given_m, x, y):
    totalError = np.sum(np.power(y - (given_m * x + given_b),2))
    return totalError / ( float(len(y)) * 2 )


def step_gradient(given_b, given_m, given_x, given_y, given_learning_rate):
    N = float(given_y.shape[1])
    tmp = given_y - ((given_m * given_x) + given_b)
    # tmp = np.power(tmp, 2)
    b_gradient = -(1 / N) * np.sum(tmp)
    m_gradient = -(1 / N) * tmp * given_x.T
    current_b = given_b - (given_learning_rate * b_gradient)
    current_m = given_m - (given_learning_rate * m_gradient[0,0])
    return current_b, current_m


def univariate_regression(given_x, given_y, given_b, given_m, given_learning_rate, given_num_of_iter):
    current_b = given_b
    current_m = given_m

    plt.ion()

    fig1 = plt.figure(1)
    ax = fig1.add_subplot(111)
    plt.axis([0, given_x.max(), 0, given_y.max()])
    plt.xlabel('CRIM')
    plt.ylabel('MEDV')
    x_arr = np.array(given_x)[0]
    y_arr = np.array(given_y)[0]
    _, line = ax.plot(x_arr, y_arr, 'bo', x_arr, current_m * x_arr + current_b, 'r')

    errors = [compute_error_for_line_given_points(given_b, given_m, given_x, given_y)]
    for i in range(given_num_of_iter):
        current_b, current_m = step_gradient(current_b, current_m, given_x, given_y, given_learning_rate)

        if i % 50 == 0:
            line.set_ydata(current_m * x_arr + current_b)
            fig1.canvas.draw()
            plt.pause(0.01)
            print(i)

        errors.append(compute_error_for_line_given_points(current_b, current_m, given_x, given_y))
    return current_b, current_m, errors


if __name__ == "__main__":
    inputs = read_file()
    x1 = np.matrix([x[COLNUM[0]] for x in inputs])
    x2 = np.matrix([x[COLNUM[1]] for x in inputs])
    y = np.matrix([x[COLNUM[2]] for x in inputs])
    m = b = 1000
    learning_rate = 0.001
    num_of_iter = 10000
    # plt.plot(x2, y, 'bo')
    # plt.axis([0, x1.max(), 0, y.max()])
    # plt.ylabel('MEDV')
    # plt.xlabel('CRIM')
    # plt.show()
    b, m, errors = univariate_regression(x1, y, b, m, learning_rate, num_of_iter)
    # b, m = univariate_regression(x2, y, b, m, learning_rate, num_of_iter)

    plt.figure(2)
    plt.axis([0, num_of_iter, 0, errors[0]])
    plt.xlabel('iter')
    plt.ylabel('err')
    plt.plot(errors)
    plt.show()
    plt.pause(10000)
    err = compute_error_for_line_given_points(b, m, x1, y)
    print(err)
