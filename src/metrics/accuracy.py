from __future__ import annotations

import random as rd

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

from ..models import binary_lda as ml

NB_MULTIPLE_IMPUTATION = 10


def _weighted_distance(
    x1: np.ndarray, x2: np.ndarray, var: np.ndarray
) -> float:
    total = 0.0
    for i in range(len(x1)):
        if x1[i] == 0 or x2[i] == 0:
            total += 1
        else:
            total += ((x1[i] - x2[i]) ** 2) / (2 * var[i])
    total = np.sqrt(total)
    return total


def _majority_vote(predictions_list: list[np.ndarray]) -> np.ndarray:
    dim = len(predictions_list[0])
    result = np.zeros(dim)
    for i in range(dim):
        for j in range(len(predictions_list)):
            result[i] += predictions_list[j][i]
        if result[i] > len(predictions_list) / 2:
            result[i] = 1
        else:
            result[i] = 0
    return result


def _compute_feature_mean(
    data: np.ndarray, dim: int, n_samples: int
) -> np.ndarray:
    mean = np.zeros(dim)
    for j in range(dim):
        n = 0
        for i in range(n_samples):
            val = data[i][j]
            if val != 0:
                n += 1
                mean[j] += val
        mean[j] /= n
    return mean


def _compute_feature_variance(
    data: np.ndarray, mean: np.ndarray, dim: int, n_samples: int
) -> np.ndarray:
    var = np.zeros(dim)
    for j in range(dim):
        n = 0
        for i in range(n_samples):
            val = data[i][j]
            if val != 0:
                n += 1
                var[j] += (val - mean[j]) ** 2
        var[j] /= n
    return var


def _impute_closest_single(
    data: np.ndarray,
    labels: np.ndarray | None,
    var: np.ndarray,
    n_samples: int,
    dim: int,
    use_labels: bool,
) -> np.ndarray:
    imputed = np.copy(data)
    for i in range(n_samples):
        for j in range(dim):
            if imputed[i][j] == 0:
                min_dis = dim + 1
                closest_index = 0
                for _ in range(100):
                    rand_idx = rd.randint(0, n_samples - 1)
                    if imputed[rand_idx][j] != 0:
                        if not use_labels or labels[rand_idx] == labels[i]:
                            dis = _weighted_distance(
                                imputed[i], imputed[rand_idx], var
                            )
                            if dis < min_dis:
                                closest_index = rand_idx
                                min_dis = dis
                imputed[i][j] = imputed[closest_index][j]
    return imputed


def _impute_closest_double(
    data: np.ndarray,
    labels: np.ndarray | None,
    var: np.ndarray,
    n_samples: int,
    dim: int,
    use_labels: bool,
) -> tuple[np.ndarray, np.ndarray]:
    imputed_first = np.copy(data)
    imputed_second = np.copy(data)
    for i in range(n_samples):
        for j in range(dim):
            if imputed_first[i][j] == 0:
                min_dis1 = dim + 1
                min_dis2 = dim + 1
                closest_index1 = 0
                closest_index2 = 0
                for _ in range(100):
                    rand_idx = rd.randint(0, n_samples - 1)
                    if imputed_first[rand_idx][j] != 0:
                        if not use_labels or labels[rand_idx] == labels[i]:
                            dis = _weighted_distance(
                                imputed_first[i], imputed_first[rand_idx], var
                            )
                            if dis < min_dis1:
                                closest_index1 = rand_idx
                                min_dis1 = dis
                            elif dis < min_dis2:
                                closest_index2 = rand_idx
                                min_dis2 = dis
                imputed_first[i][j] = imputed_first[closest_index1][j]
                imputed_second[i][j] = imputed_second[closest_index2][j]
    return imputed_first, imputed_second


def _acc_multiple_closest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    dim = len(x_train[0])
    len_training = len(x_train)
    len_testing = len(x_test)

    mean = _compute_feature_mean(x_train, dim, len_training)
    var = _compute_feature_variance(x_train, mean, dim, len_training)

    x_train_first, x_train_second = _impute_closest_double(
        x_train, y_train, var, len_training, dim, use_labels=True
    )
    x_test_first, x_test_second = _impute_closest_double(
        x_test, None, var, len_testing, dim, use_labels=False
    )

    predictions = []
    for i in range(NB_MULTIPLE_IMPUTATION + 1):
        x_train_interp = (
            i * x_train_first + (NB_MULTIPLE_IMPUTATION - i) * x_train_second
        ) / NB_MULTIPLE_IMPUTATION
        x_test_interp = (
            i * x_test_first + (NB_MULTIPLE_IMPUTATION - i) * x_test_second
        ) / NB_MULTIPLE_IMPUTATION
        lda = LinearDiscriminantAnalysis()
        lda.fit(x_train_interp, y_train)
        predictions.append(lda.predict(x_test_interp))

    sol = _majority_vote(predictions)
    return accuracy_score(y_test, sol)


def _acc_no_imputation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    return lda.score(x_test, y_test)


def _acc_grand_mean(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    dim = len(x_train[0])
    len_training = len(x_train)
    len_testing = len(x_test)

    total_count = 0
    grand_mean = np.zeros(dim)
    for i in range(len_training):
        for j in range(dim):
            val = x_train[i][j]
            if val != 0:
                total_count += 1
                grand_mean[j] += val
    grand_mean /= total_count

    x_train_imputed = np.copy(x_train)
    for i in range(len_training):
        for j in range(dim):
            if x_train_imputed[i][j] == 0:
                x_train_imputed[i][j] = grand_mean[j]

    x_test_imputed = np.copy(x_test)
    for i in range(len_testing):
        for j in range(dim):
            if x_test_imputed[i][j] == 0:
                x_test_imputed[i][j] = grand_mean[j]

    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train_imputed, y_train)
    return lda.score(x_test_imputed, y_test)


def _acc_conditional_mean(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    dim = len(x_train[0])
    len_training = len(x_train)
    len_testing = len(x_test)

    count_0 = 0
    mean_0 = np.zeros(dim)
    count_1 = 0
    mean_1 = np.zeros(dim)

    for i in range(len_training):
        if y_train[i] == 0:
            for j in range(dim):
                val = x_train[i][j]
                if val != 0:
                    count_0 += 1
                    mean_0[j] += val
        else:
            for j in range(dim):
                val = x_train[i][j]
                if val != 0:
                    count_1 += 1
                    mean_1[j] += val

    for j in range(dim):
        mean_0[j] = mean_0[j] / count_0
    for j in range(dim):
        mean_1[j] = mean_1[j] / count_1

    x_train_imputed = np.copy(x_train)
    for i in range(len_training):
        for j in range(dim):
            if x_train_imputed[i][j] == 0:
                if y_train[i] == 0:
                    x_train_imputed[i][j] = mean_0[j]
                if y_train[i] == 1:
                    x_train_imputed[i][j] = mean_1[j]

    x_test_class0 = np.copy(x_test)
    for i in range(len_testing):
        for j in range(dim):
            if x_test_class0[i][j] == 0:
                x_test_class0[i][j] = mean_0[j]

    x_test_class1 = np.copy(x_test)
    for i in range(len_testing):
        for j in range(dim):
            if x_test_class1[i][j] == 0:
                x_test_class1[i][j] = mean_1[j]

    # Conditional mean imputation fills missing values with per-class means,
    # so test samples get two different feature matrices (one per assumed class).
    # BinaryLDA scores each class's matrix separately, unlike sklearn's LDA.
    lda = ml.BinaryLDA(
        n_components=None,
        priors=None,
        shrinkage=None,
        solver="eigen",
        store_covariance=False,
        tol=0.0001,
    )
    lda.fit(x_train_imputed, y_train)
    return lda.score(x_test_class0, x_test_class1, y_test)


def _acc_closest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    dim = len(x_train[0])
    len_training = len(x_train)
    len_testing = len(x_test)

    mean = _compute_feature_mean(x_train, dim, len_training)
    var = _compute_feature_variance(x_train, mean, dim, len_training)

    x_train_imputed = _impute_closest_single(
        x_train, y_train, var, len_training, dim, use_labels=True
    )
    x_test_imputed = _impute_closest_single(
        x_test, None, var, len_testing, dim, use_labels=False
    )

    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train_imputed, y_train)
    return lda.score(x_test_imputed, y_test)


def _acc_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    dim = len(x_train[0])
    len_training = len(x_train)
    len_testing = len(x_test)

    x_train_imputed = np.copy(x_train)
    x_test_imputed = np.copy(x_test)

    for j in range(1, dim):
        features = []
        targets = []
        for h in range(len_training):
            if x_train_imputed[h][j] != 0:
                features.append(x_train_imputed[h][0:j])
                targets.append(x_train_imputed[h][j])
        reg = LinearRegression().fit(features, targets)

        for h in range(len_training):
            if x_train_imputed[h][j] == 0:
                x_train_imputed[h][j] = reg.predict([x_train_imputed[h][0:j]])[0]
        for h in range(len_testing):
            if x_test_imputed[h][j] == 0:
                x_test_imputed[h][j] = reg.predict([x_test_imputed[h][0:j]])[0]

    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train_imputed, y_train)
    return lda.score(x_test_imputed, y_test)


def _acc_multiple_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    dim = len(x_train[0])
    len_training = len(x_train)
    len_testing = len(x_test)

    mean = _compute_feature_mean(x_train, dim, len_training)
    var = _compute_feature_variance(x_train, mean, dim, len_training)

    results = []
    for _ in range(NB_MULTIPLE_IMPUTATION):
        x_train_imputed = np.copy(x_train)
        x_test_imputed = np.copy(x_test)

        for j in range(1, dim):
            features = []
            targets = []
            for h in range(len_training):
                if x_train_imputed[h][j] != 0:
                    features.append(x_train_imputed[h][0:j])
                    targets.append(x_train_imputed[h][j])
            reg = LinearRegression().fit(features, targets)

            for h in range(len_training):
                if x_train_imputed[h][j] == 0:
                    x_train_imputed[h][j] = reg.predict(
                        [x_train_imputed[h][0:j]]
                    )[0] + np.random.normal(0, np.sqrt(var[j]))
            for h in range(len_testing):
                if x_test_imputed[h][j] == 0:
                    x_test_imputed[h][j] = reg.predict(
                        [x_test_imputed[h][0:j]]
                    )[0] + np.random.normal(0, np.sqrt(var[j]))

        lda = LinearDiscriminantAnalysis()
        lda.fit(x_train_imputed, y_train)
        results.append(lda.predict(x_test_imputed))

    sol = _majority_vote(results)
    return accuracy_score(y_test, sol)


IMPUTATION_METHODS = {
    "multiple_closest": _acc_multiple_closest,
    "no_imputation": _acc_no_imputation,
    "grand_mean": _acc_grand_mean,
    "conditional_mean": _acc_conditional_mean,
    "closest": _acc_closest,
    "regression": _acc_regression,
    "multiple_regression": _acc_multiple_regression,
}


def acc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    imputation_method: str,
) -> float:
    method_func = IMPUTATION_METHODS.get(imputation_method)
    if method_func is None:
        raise ValueError(f"Unknown imputation method: {imputation_method}")
    return method_func(x_train, y_train, x_test, y_test)
