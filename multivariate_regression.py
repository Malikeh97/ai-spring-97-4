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


def compute_error(b, m, x, y):
    totalError = np.sum(np.power(y - (m * x + b),2))
    return totalError / ( float(len(y)) * 2 )


def step_gradient(init_B, init_x, init_y, lrate):
    N = float(init_y.shape[1])
    tmp = init_y - (init_B.T * init_x.T)
    B_gradient = -(1 / N) * tmp * init_x
    print("---")
    print(B_gradient)
    new_B = init_B - (lrate * B_gradient.T)
    return new_B


def multivariate_regression(init_x, init_y, lrate, iters):
    # plt.ion()
    # fig1 = plt.figure(1)
    # ax = fig1.add_subplot(111)
    # plt.axis([0, x.max(), 0, given_y.max()])
    # plt.xlabel('CRIM')
    # plt.ylabel('MEDV')
    # x_arr = np.array(x)[0]
    # y_arr = np.array(given_y)[0]
    # _, line = ax.plot(x_arr, y_arr, 'bo', x_arr, m * x_arr + b, 'r')

    # errors = [compute_error_for_line_given_points(b, m, x, given_y)]
    new_B = np.zeros((14,1))
    for i in range(20):
        new_B = step_gradient(new_B, init_x, init_y, lrate)
        # if i % 50 == 0:
        #     line.set_ydata(m * x_arr + b)
        #     fig1.canvas.draw()
        #     plt.pause(0.01)
        #     print(i)

        # errors.append(compute_error_for_line_given_points(b, m, x, given_y))
    return new_B


if __name__ == "__main__":
    inputs = read_file()
    X = np.ones((506,1))
    X = np.append(X, np.matrix([x[0:13] for x in inputs]), axis=1)
    Y = np.matrix([x[13] for x in inputs])
    learning_rate = 0.001
    num_of_iter = 7000
    # plt.plot(x2, y, 'bo')
    # plt.axis([0, x1.max(), 0, y.max()])
    # plt.ylabel('MEDV')
    # plt.xlabel('CRIM')
    # plt.show()
    B = multivariate_regression(X, Y, learning_rate, num_of_iter)
    print(B)



