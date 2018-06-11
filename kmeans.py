import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

np.seterr(all='raise')


def read_file():
    data = pd.read_csv('cars.csv')
    data.head()

    mpg = normalize(data['mpg'].values)
    cubicinches = normalize(data['cubicinches'].values)
    hp = normalize(data['hp'].values)
    weightlbs = normalize(data['weightlbs'].values)
    time_to_60 = normalize(data['time-to-60'].values)
    year = normalize(data['year'].values)

    return np.array(list(zip(mpg, cubicinches, hp, weightlbs, time_to_60, year)))


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
    X = read_file()

    errors = []
    maxk = 20
    for k in range(1, maxk):
        # k = 2
        # C = initial_centroids(X, k)
        print(k)
        C = X[:k, :]
        # Cluster Labels (0, 1, 2, ..., k - 1)
        clusters = np.zeros(len(X), dtype=np.int)
        for m in range(200):
            for i in range(len(X)):
                distances = dist(X[i], C)
                clusters[i] = np.argmin(distances)

            for i in range(k):
                points = [X[j] for j in range(len(X)) if clusters[j] == i]
                C[i] = np.mean(points, axis=0)

        errors.append(compute_error(C, X, clusters))

    plt.axis([0, maxk, 0, errors[0] + 1])
    plt.plot([x for x in range(1, maxk)], errors, 'o-')
    plt.show()


if __name__ == "__main__":
    main()
