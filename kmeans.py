import numpy as np
from matplotlib import pyplot as plt

np.seterr(all='raise')


def read_file():
    with open('cars.csv') as f:
        content = f.readlines()
    content = [x.strip().split(',') for x in content]
    del content[0]
    missing = []
    for i in range(len(content)):
        for j in range(len(content[i])):
            try:
                content[i][j] = float(content[i][j])
            except ValueError:
                content[i][j] = 0.0
                missing.append((i, j))

    return content, missing


def initial_centroids(X, k):
    C = np.array([np.random.uniform(np.min(X), np.max(X), size=k)]).T
    for i in range(X.shape[1] - 1):
        C = np.append(C, np.array([np.random.uniform(np.min(X), np.max(X), size=k)]).T, axis=1)
    return C


def normalize(arr):
    return (arr - np.mean(arr, axis=0)) / np.std(arr, axis=0)


def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


def compute_error(C, X, clusters):
    sum = 0
    for i in range(len(X)):
        sum += np.sum(np.power(X[i] - C[clusters[i]], 2))
    return sum / len(X)


def main():
    inputs, missing = read_file()
    inputs = np.array(inputs)

    cols_average = np.mean(inputs, axis=0)
    for m in missing:
        inputs[m[0], m[1]] = cols_average[m[1]]

    inputs = ((inputs - cols_average) / np.std(inputs, axis=0)).astype(np.float128)

    errors = []
    max_k = 20
    for k in range(1, max_k):
        # k = 2
        # C = initial_centroids(X, k)
        print(k)
        C = inputs[:k, :]
        # a = np.random.randint(inputs.shape[0], size=k)
        # C = inputs[a, :]
        # print(a)
        # Cluster Labels (0, 1, 2, ..., k - 1)
        clusters = np.zeros(len(inputs), dtype=np.int)
        for m in range(200):
            for i in range(len(inputs)):
                distances = dist(inputs[i], C)
                clusters[i] = np.argmin(distances)

            for i in range(k):
                points = [inputs[j] for j in range(len(inputs)) if clusters[j] == i]
                C[i] = np.mean(points, axis=0)

        errors.append(compute_error(C, inputs, clusters))

    plt.axis([0, max_k, 0, errors[0] + 1])
    plt.plot([x for x in range(1, max_k)], errors, 'o-')
    plt.show()


if __name__ == "__main__":
    main()
