from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import empirical_covariance, ledoit_wolf, shrunk_covariance
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array, check_X_y
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted


def _cov(x: np.ndarray, shrinkage: str | float | None = None) -> np.ndarray:
    shrinkage = "empirical" if shrinkage is None else shrinkage
    if isinstance(shrinkage, str):
        if shrinkage == "auto":
            sc = StandardScaler()
            x = sc.fit_transform(x)
            s = ledoit_wolf(x)[0]
            s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
        elif shrinkage == "empirical":
            s = empirical_covariance(x)
        else:
            raise ValueError("unknown shrinkage parameter")
    elif isinstance(shrinkage, (float, int)):
        if shrinkage < 0 or shrinkage > 1:
            raise ValueError("shrinkage parameter must be between 0 and 1")
        s = shrunk_covariance(empirical_covariance(x), shrinkage)
    else:
        raise TypeError("shrinkage must be of string or int type")
    return s


def _class_means(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), x.shape[1]))
    np.add.at(means, y, x)
    means /= cnt[:, None]
    return means


def _class_cov(
    x: np.ndarray,
    y: np.ndarray,
    priors: np.ndarray,
    shrinkage: str | float | None = None,
) -> np.ndarray:
    classes = np.unique(y)
    cov = np.zeros(shape=(x.shape[1], x.shape[1]))
    for idx, group in enumerate(classes):
        x_group = x[y == group, :]
        cov += priors[idx] * np.atleast_2d(_cov(x_group, shrinkage))
    return cov


class BinaryLDA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        solver: str = "svd",
        shrinkage: str | float | None = None,
        priors: np.ndarray | None = None,
        n_components: int | None = None,
        store_covariance: bool = False,
        tol: float = 1e-4,
    ):
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance
        self.tol = tol

    def _solve_lsqr(
        self, x: np.ndarray, y: np.ndarray, shrinkage: str | float | None
    ) -> None:
        self.means_ = _class_means(x, y)
        self.covariance_ = _class_cov(x, y, self.priors_, shrinkage)
        self.coef_ = linalg.lstsq(self.covariance_, self.means_.T)[0].T
        self.intercept_ = -0.5 * np.diag(
            np.dot(self.means_, self.coef_.T)
        ) + np.log(self.priors_)

    def _solve_eigen(
        self, x: np.ndarray, y: np.ndarray, shrinkage: str | float | None
    ) -> None:
        self.means_ = _class_means(x, y)
        self.covariance_ = _class_cov(x, y, self.priors_, shrinkage)

        sw = self.covariance_
        st = _cov(x, shrinkage)
        sb = st - sw

        evals, evecs = linalg.eigh(sb, sw)
        self.explained_variance_ratio_ = np.sort(evals / np.sum(evals))[::-1][
            : self._max_components
        ]
        evecs = evecs[:, np.argsort(evals)[::-1]]

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(
            np.dot(self.means_, self.coef_.T)
        ) + np.log(self.priors_)

    def _solve_svd(self, x: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = x.shape
        n_classes = len(self.classes_)

        self.means_ = _class_means(x, y)
        if self.store_covariance:
            self.covariance_ = _class_cov(x, y, self.priors_)

        xc = []
        for idx, group in enumerate(self.classes_):
            x_group = x[y == group, :]
            xc.append(x_group - self.means_[idx])

        self.xbar_ = np.dot(self.priors_, self.means_)

        xc = np.concatenate(xc, axis=0)

        std = xc.std(axis=0)
        std[std == 0] = 1.0
        fac = 1.0 / (n_samples - n_classes)

        x_scaled = np.sqrt(fac) * (xc / std)
        u, s, v = linalg.svd(x_scaled, full_matrices=False)

        rank = np.sum(s > self.tol)
        if rank < n_features:
            warnings.warn("Variables are collinear.")

        scalings = (v[:rank] / std).T / s[:rank]

        x_between = np.dot(
            (
                (np.sqrt((n_samples * self.priors_) * fac))
                * (self.means_ - self.xbar_).T
            ).T,
            scalings,
        )
        _, s, v = linalg.svd(x_between, full_matrices=0)

        self.explained_variance_ratio_ = (s**2 / np.sum(s**2))[
            : self._max_components
        ]
        rank = np.sum(s > self.tol * s[0])
        self.scalings_ = np.dot(scalings, v.T[:, :rank])
        coef = np.dot(self.means_ - self.xbar_, self.scalings_)
        self.intercept_ = -0.5 * np.sum(coef**2, axis=1) + np.log(self.priors_)
        self.coef_ = np.dot(coef, self.scalings_.T)
        self.intercept_ -= np.dot(self.xbar_, self.coef_.T)

    def fit(
        self, x: np.ndarray, y: np.ndarray
    ) -> "BinaryLDA":
        x, y = check_X_y(
            x, y, ensure_min_samples=2, estimator=self, dtype=[np.float64, np.float32]
        )
        self.classes_ = unique_labels(y)
        n_samples, _ = x.shape
        n_classes = len(self.classes_)

        if n_samples == n_classes:
            raise ValueError(
                "The number of samples must be more than the number of classes."
            )

        if self.priors is None:
            _, y_t = np.unique(y, return_inverse=True)
            self.priors_ = np.bincount(y_t) / float(len(y))
        else:
            self.priors_ = np.asarray(self.priors)

        if (self.priors_ < 0).any():
            raise ValueError("priors must be non-negative")
        if not np.isclose(self.priors_.sum(), 1.0):
            warnings.warn("The priors do not sum to 1. Renormalizing", UserWarning)
            self.priors_ = self.priors_ / self.priors_.sum()

        max_components = min(len(self.classes_) - 1, x.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                warnings.warn(
                    "n_components cannot be larger than min(n_features, "
                    "n_classes - 1). Using min(n_features, "
                    "n_classes - 1) = min(%d, %d - 1) = %d components."
                    % (x.shape[1], len(self.classes_), max_components),
                    FutureWarning,
                )
                self._max_components = max_components
            else:
                self._max_components = self.n_components

        if self.solver == "svd":
            if self.shrinkage is not None:
                raise NotImplementedError("shrinkage not supported")
            self._solve_svd(x, y)
        elif self.solver == "lsqr":
            self._solve_lsqr(x, y, shrinkage=self.shrinkage)
        elif self.solver == "eigen":
            self._solve_eigen(x, y, shrinkage=self.shrinkage)
        else:
            raise ValueError(
                "unknown solver {} (valid solvers are 'svd', "
                "'lsqr', and 'eigen').".format(self.solver)
            )

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.solver == "lsqr":
            raise NotImplementedError(
                "transform not implemented for 'lsqr' solver (use 'svd' or 'eigen')."
            )
        check_is_fitted(self, ["xbar_", "scalings_"], all_or_any=any)

        x = check_array(x)
        if self.solver == "svd":
            x_new = np.dot(x - self.xbar_, self.scalings_)
        elif self.solver == "eigen":
            x_new = np.dot(x, self.scalings_)

        return x_new[:, : self._max_components]

    def decision_function(
        self, x_class0: np.ndarray, x_class1: np.ndarray
    ) -> np.ndarray:
        # Each class gets its own feature matrix because conditional mean
        # imputation fills missing values with the class-specific mean,
        # producing different feature values for the same sample depending
        # on the assumed class.
        n_features = self.coef_.shape[1]
        if x_class0.shape[1] != n_features:
            raise ValueError(
                "x_class0 has %d features per sample; expecting %d"
                % (x_class0.shape[1], n_features)
            )

        score_0 = (
            safe_sparse_dot(x_class0, self.coef_[0, :].T, dense_output=True)
            + self.intercept_[0]
        )
        score_1 = (
            safe_sparse_dot(x_class1, self.coef_[1, :].T, dense_output=True)
            + self.intercept_[1]
        )

        return score_1 - score_0

    def predict(
        self, x_class0: np.ndarray, x_class1: np.ndarray
    ) -> np.ndarray:
        scores = self.decision_function(x_class0, x_class1)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int32)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def score(
        self,
        x_class0: np.ndarray,
        x_class1: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> float:
        return accuracy_score(
            y, self.predict(x_class0, x_class1), sample_weight=sample_weight
        )
