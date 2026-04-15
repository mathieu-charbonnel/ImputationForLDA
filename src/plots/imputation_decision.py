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
    optimal = np.zeros(num_sizes)

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
            best_score = 0.0

            r1 = acc.acc(
                full_x_train[0:size, :],
                full_y_train[0:size],
                x_test,
                y_test,
                "no_imputation",
            )
            r2 = acc.acc(
                full_x_train[0:size, :],
                full_y_train[0:size],
                x_test,
                y_test,
                "grand_mean",
            )
            r3 = acc.acc(
                full_x_train[0:size, :],
                full_y_train[0:size],
                x_test,
                y_test,
                "conditional_mean",
            )
            r4 = acc.acc(
                full_x_train[0:size, :],
                full_y_train[0:size],
                x_test,
                y_test,
                "closest",
            )
            r5 = acc.acc(
                full_x_train[0:size, :],
                full_y_train[0:size],
                x_test,
                y_test,
                "regression",
            )

            without_removing[j] += acc.acc(
                x_train_intact[0:size][:],
                full_y_train[0:size],
                x_test_intact,
                y_test,
                "no_imputation",
            )
            without_imp[j] += r1
            grand_mean[j] += r2
            conditional_mean[j] += r3
            closest[j] += r4
            regression[j] += r5

            train_score_no_imp = acc.acc(
                full_x_train[0:size][:],
                full_y_train[0:size],
                full_x_train[0:size][:],
                full_y_train[0:size],
                "no_imputation",
            )
            if train_score_no_imp > best_score:
                best_score = r1

            train_score_grand = acc.acc(
                full_x_train[0:size][:],
                full_y_train[0:size],
                full_x_train[0:size][:],
                full_y_train[0:size],
                "grand_mean",
            )
            if train_score_grand > best_score:
                best_score = r2

            train_score_cond = acc.acc(
                full_x_train[0:size][:],
                full_y_train[0:size],
                full_x_train[0:size][:],
                full_y_train[0:size],
                "conditional_mean",
            )
            if train_score_cond > best_score:
                best_score = r3

            train_score_closest = acc.acc(
                full_x_train[0:size][:],
                full_y_train[0:size],
                full_x_train[0:size][:],
                full_y_train[0:size],
                "closest",
            )
            if train_score_closest > best_score:
                best_score = r4

            train_score_reg = acc.acc(
                full_x_train[0:size][:],
                full_y_train[0:size],
                full_x_train[0:size][:],
                full_y_train[0:size],
                "regression",
            )
            if train_score_reg > best_score:
                best_score = r5

            optimal[j] += best_score

    without_removing /= NB_TESTS
    without_imp /= NB_TESTS
    grand_mean /= NB_TESTS
    conditional_mean /= NB_TESTS
    closest /= NB_TESTS
    regression /= NB_TESTS
    optimal /= NB_TESTS

    plt.figure(figsize=(10, 5))
    plt.plot(TRAINING_SET_LABELS, optimal, color="yellow", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, without_removing, color="purple", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, without_imp, color="green", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, grand_mean, color="blue", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, conditional_mean, color="orange", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, closest, color="red", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, regression, color="black", linewidth=2)

    labels = [
        "optimal",
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
    plt.savefig(
        "opt"
        + "Covariance: "
        + covariance_type
        + "  Dim: "
        + str(dim)
        + "  Type missingness: "
        + type_missingness
        + "  Prob_missingness: "
        + str(prob_missingness)
        + ".jpg"
    )
    plt.show()
