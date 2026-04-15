from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas

from ..metrics import accuracy as acc

NB_TESTS = 10
TRAINING_SET_SIZES = [50, 100, 250, 500, 1000]
TRAINING_SET_LABELS = ["50", "100", "250", "500", "1000"]


def one_graph(prob_missingness: float) -> None:
    num_sizes = len(TRAINING_SET_SIZES)
    without_imp = np.zeros(num_sizes)
    grand_mean = np.zeros(num_sizes)
    conditional_mean = np.zeros(num_sizes)
    closest = np.zeros(num_sizes)
    regression = np.zeros(num_sizes)
    without_removing = np.zeros(num_sizes)
    multiple_closest = np.zeros(num_sizes)
    multiple_regression = np.zeros(num_sizes)

    dini = pandas.read_csv("data_banknote_authentication.csv").to_numpy()
    np.random.shuffle(dini)
    x_train_intact = dini[0:1000, 0:4]
    x_test_intact = dini[1000:, 0:4]
    y_train_intact = dini[0:1000, 4]
    y_test_intact = dini[1000:, 4]

    for _ in range(NB_TESTS):
        df = pandas.read_csv("data_banknote_authentication.csv").to_numpy()
        np.random.shuffle(df)
        for i in range(1371):
            for j in range(4):
                if (np.random.uniform() < prob_missingness * 2) and (
                    df[i, 4] == 1.0
                ):
                    df[i, j] = 0
        full_x_train = df[0:1000, 0:4]
        full_y_train = df[0:1000, 4]
        x_test = df[1000:, 0:4]
        y_test = df[1000:, 4]

        for j in range(num_sizes):
            size = TRAINING_SET_SIZES[j]
            without_removing[j] += acc.acc(
                x_train_intact[0:size][:],
                y_train_intact[0:size],
                x_test_intact,
                y_test_intact,
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
            multiple_closest[j] += acc.acc(
                full_x_train[0:size, :],
                full_y_train[0:size],
                x_test,
                y_test,
                "multiple_closest",
            )
            multiple_regression[j] += acc.acc(
                full_x_train[0:size, :],
                full_y_train[0:size],
                x_test,
                y_test,
                "multiple_regression",
            )

    without_removing /= NB_TESTS
    without_imp /= NB_TESTS
    grand_mean /= NB_TESTS
    conditional_mean /= NB_TESTS
    closest /= NB_TESTS
    regression /= NB_TESTS
    multiple_closest /= NB_TESTS
    multiple_regression /= NB_TESTS

    plt.figure(figsize=(10, 5))
    plt.plot(TRAINING_SET_LABELS, without_removing, color="purple", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, without_imp, color="green", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, grand_mean, color="blue", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, conditional_mean, color="orange", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, closest, color="red", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, regression, color="black", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, multiple_closest, color="pink", linewidth=2)
    plt.plot(TRAINING_SET_LABELS, multiple_regression, color="grey", linewidth=2)

    labels = [
        "without_removing",
        "No imputation",
        "Grand Mean",
        "Conditional Mean",
        "Closest neighbour",
        "Regression",
        "multiple_closest",
        "multiple_regression",
    ]

    plt.title(
        "  Real_life_data MNAR   Prob_missingness: " + str(prob_missingness)
    )
    plt.legend(labels=labels)
    plt.ylabel("Accuracy", fontsize=10)
    plt.xlabel("Training size", fontsize=10)
    plt.show()
