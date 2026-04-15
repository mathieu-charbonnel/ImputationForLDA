from __future__ import annotations

import numpy as np
from sklearn.datasets import make_spd_matrix

RATIO = 0.5


def data_generation(
    covariance_type: str,
    dim: int,
    len_training: int,
    len_testing: int,
) -> list[list]:
    mean_0 = 2 * np.random.rand(dim) - 0.5 * np.ones(dim)
    mean_1 = 2 * np.random.rand(dim) - 0.5 * np.ones(dim)

    cov = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if i == j:
                cov[i][j] = 1
            else:
                cov[i][j] = np.exp(-np.abs(i - j))

    if covariance_type == "str_correlation_higherIndex":
        for i in range(dim):
            for j in range(dim):
                cov[i][j] = cov[i][j] * max(i, j)

    if covariance_type == "str_correlation+high_diagonal":
        for i in range(dim):
            cov[i][i] *= 5

    if covariance_type == "random":
        cov = make_spd_matrix(dim, random_state=None)

    x_training: list[np.ndarray] = []
    y_training: list[int] = []
    for _ in range(len_training):
        r = np.random.uniform()
        if r < RATIO:
            x_training.append(np.random.multivariate_normal(mean_0, cov))
            y_training.append(0)
        else:
            x_training.append(np.random.multivariate_normal(mean_1, cov))
            y_training.append(1)

    x_testing: list[np.ndarray] = []
    y_testing: list[int] = []
    for _ in range(len_testing):
        r = np.random.uniform()
        if r < RATIO:
            x_testing.append(np.random.multivariate_normal(mean_0, cov))
            y_testing.append(0)
        else:
            x_testing.append(np.random.multivariate_normal(mean_1, cov))
            y_testing.append(1)

    return [x_training, y_training, x_testing, y_testing]


def removing_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    type_missingness: str,
    prob_missingness: float,
) -> list[np.ndarray]:
    p = prob_missingness
    dim = len(x_train[0])
    len_train = len(x_train)
    len_test = len(x_test)
    x_train_missing = np.copy(x_train)
    x_test_missing = np.copy(x_test)

    if type_missingness == "MCAR":
        for i in range(len_train):
            for j in range(1, dim):
                if np.random.uniform() < p:
                    x_train_missing[i][j] = 0
        for i in range(len_test):
            for j in range(1, dim):
                if np.random.uniform() < p:
                    x_test_missing[i][j] = 0

    if type_missingness == "MAR":
        for j in range(1, dim):
            target_ratio = p
            ratio_missing = 1
            limit = 0
            while ratio_missing > target_ratio:
                limit += 0.05
                ratio_missing = 0
                for i in range(len_train):
                    if np.abs(x_train_missing[i][j - 1]) > limit:
                        ratio_missing += 1 / len_train

            for i in range(len_train):
                if np.abs(x_train_missing[i][j - 1]) > limit:
                    x_train_missing[i][j] = 0
            for i in range(len_test):
                if np.abs(x_test_missing[i][j - 1]) > limit:
                    x_test_missing[i][j] = 0

    if type_missingness == "MNAR":
        for i in range(len_train):
            if y_train[i] == 0:
                for j in range(1, dim):
                    if np.random.uniform() < p * (1 / RATIO):
                        x_train_missing[i][j] = 0
        for i in range(len_test):
            if y_test[i] == 0:
                for j in range(1, dim):
                    if np.random.uniform() < p * (1 / RATIO):
                        x_test_missing[i][j] = 0

    return [x_train_missing, y_train, x_test_missing, y_test, x_train, x_test]


def full_gen(
    covariance_type: str,
    dim: int,
    len_training: int,
    len_testing: int,
    type_missingness: str,
    prob_missingness: float,
) -> list[np.ndarray]:
    generated = data_generation(covariance_type, dim, len_training, len_testing)
    return removing_data(
        generated[0],
        generated[1],
        generated[2],
        generated[3],
        type_missingness,
        prob_missingness,
    )
