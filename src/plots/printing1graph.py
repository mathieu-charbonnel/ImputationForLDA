from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from ..loading import datagen as dt
from ..metrics import accuracy as acc

NB_TESTS = 10
TRAINING_SET_SIZES = [50, 100, 250, 500, 1000, 2000]
TRAINING_SET_LABELS = ["50", "100", "250", "500", "1000", "2000"]


def one_graph(
    covariance_type: str,
    dim: int,
    type_missingness: str,
    prob_missingness: float,
) -> None:
    num_sizes = len(TRAINING_SET_SIZES)
    without_imp = np.zeros(num_sizes)
    grand_mean = np.zeros(num_sizes)
    conditional_mean = np.zeros(num_sizes)
    closest = np.zeros(num_sizes)
    regression = np.zeros(num_sizes)
    without_removing = np.zeros(num_sizes)

    for _ in range(NB_TESTS):
        data = dt.full_gen(
            covariance_type, dim, 2000, 1000, type_missingness, prob_missingness
        )
        full_x_train = data[0]
        full_y_train = data[1]
        x_test = data[2]
        y_test = data[3]
        x_train_intact = data[4]
        x_test_intact = data[5]

        for j in range(num_sizes):
            size = TRAINING_SET_SIZES[j]
            without_removing[j] += acc.acc(
                x_train_intact[0:size][:],
                full_y_train[0:size],
                x_test_intact,
                y_test,
                "no_imputation",
            )
            without_imp[j] += acc.acc(
                full_x_train[0:size, :],
                full_y_train[0:size],
                x_test,
                y_test,
                "no_imputation",
            )
            grand_mean[j] += acc.acc(
                full_x_train[0:size, :],
                full_y_train[0:size],
                x_test,
                y_test,
                "grand_mean",
            )
            conditional_mean[j] += acc.acc(
                full_x_train[0:size, :],
                full_y_train[0:size],
                x_test,
                y_test,
                "conditional_mean",
            )
            closest[j] += acc.acc(
                full_x_train[0:size, :],
                full_y_train[0:size],
                x_test,
                y_test,
                "closest",
            )
            regression[j] += acc.acc(
                full_x_train[0:size, :],
                full_y_train[0:size],
                x_test,
                y_test,
                "regression",
            )

    without_removing /= NB_TESTS
    without_imp /= NB_TESTS
    grand_mean /= NB_TESTS
    conditional_mean /= NB_TESTS
    closest /= NB_TESTS
    regression /= NB_TESTS

    plt.figure(figsize=(10, 5))
    plt.plot(TRAINING_SET_LABELS, without_removing, color="purple", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, without_imp, color="green", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, grand_mean, color="blue", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, conditional_mean, color="orange", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, closest, color="red", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, regression, color="black", linewidth=2)

    labels = [
        "without_removing",
        "No imputation",
        "Grand Mean",
        "Conditional Mean",
        "Closest neighbour",
        "Regression",
    ]

    plt.title(
        "Covariance: "
        + covariance_type
        + "  Dim: "
        + str(dim)
        + "  Type missingness: "
        + type_missingness
        + "  Prob_missingness: "
        + str(prob_missingness)
    )
    plt.legend(labels=labels)
    plt.ylabel("Accuracy", fontsize=10)
    plt.xlabel("Training size", fontsize=10)
    plt.axis([0, 6, 0.7, 0.95])
    plt.show()
