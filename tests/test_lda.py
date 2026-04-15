import numpy as np
import pytest

from src.models.binary_lda import (
    BinaryLDA,
    _class_cov,
    _class_means,
    _cov,
)


class TestCov:
    def test_empirical_covariance(self):
        np.random.seed(42)
        x = np.random.randn(100, 3)
        result = _cov(x, shrinkage=None)
        assert result.shape == (3, 3)

    def test_auto_shrinkage(self):
        np.random.seed(42)
        x = np.random.randn(100, 3)
        result = _cov(x, shrinkage="auto")
        assert result.shape == (3, 3)

    def test_float_shrinkage(self):
        np.random.seed(42)
        x = np.random.randn(100, 3)
        result = _cov(x, shrinkage=0.5)
        assert result.shape == (3, 3)

    def test_invalid_shrinkage_string(self):
        x = np.random.randn(10, 2)
        with pytest.raises(ValueError, match="unknown shrinkage"):
            _cov(x, shrinkage="invalid")

    def test_invalid_shrinkage_range(self):
        x = np.random.randn(10, 2)
        with pytest.raises(ValueError, match="between 0 and 1"):
            _cov(x, shrinkage=1.5)

    def test_invalid_shrinkage_type(self):
        x = np.random.randn(10, 2)
        with pytest.raises(TypeError):
            _cov(x, shrinkage=[0.5])


class TestClassMeans:
    def test_two_classes(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        y = np.array([0, 0, 1, 1])
        means = _class_means(x, y)
        assert means.shape == (2, 2)
        np.testing.assert_array_almost_equal(means[0], [2.0, 3.0])
        np.testing.assert_array_almost_equal(means[1], [6.0, 7.0])


class TestClassCov:
    def test_output_shape(self):
        np.random.seed(42)
        x = np.random.randn(100, 3)
        y = np.array([0] * 50 + [1] * 50)
        priors = np.array([0.5, 0.5])
        result = _class_cov(x, y, priors)
        assert result.shape == (3, 3)


class TestBinaryLDA:
    def _make_separable_data(self):
        np.random.seed(42)
        x = np.vstack([
            np.random.randn(50, 2) + np.array([3, 3]),
            np.random.randn(50, 2) + np.array([-3, -3]),
        ])
        y = np.array([0] * 50 + [1] * 50)
        return x, y

    def test_fit_svd(self):
        x, y = self._make_separable_data()
        lda = BinaryLDA(solver="svd")
        lda.fit(x, y)
        assert hasattr(lda, "coef_")
        assert hasattr(lda, "intercept_")

    def test_fit_eigen(self):
        x, y = self._make_separable_data()
        lda = BinaryLDA(solver="eigen")
        lda.fit(x, y)
        assert hasattr(lda, "coef_")

    def test_fit_lsqr(self):
        x, y = self._make_separable_data()
        lda = BinaryLDA(solver="lsqr")
        lda.fit(x, y)
        assert hasattr(lda, "coef_")

    def test_invalid_solver(self):
        x, y = self._make_separable_data()
        lda = BinaryLDA(solver="invalid")
        with pytest.raises(ValueError, match="unknown solver"):
            lda.fit(x, y)

    def test_too_few_samples(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        lda = BinaryLDA()
        with pytest.raises(ValueError, match="number of samples"):
            lda.fit(x, y)

    def test_transform_svd(self):
        x, y = self._make_separable_data()
        lda = BinaryLDA(solver="svd")
        lda.fit(x, y)
        transformed = lda.transform(x)
        assert transformed.shape[0] == x.shape[0]
        assert transformed.shape[1] <= min(len(np.unique(y)) - 1, x.shape[1])

    def test_transform_lsqr_raises(self):
        x, y = self._make_separable_data()
        lda = BinaryLDA(solver="lsqr")
        lda.fit(x, y)
        with pytest.raises(NotImplementedError):
            lda.transform(x)

    def test_negative_priors(self):
        x, y = self._make_separable_data()
        lda = BinaryLDA(priors=[-0.5, 1.5])
        with pytest.raises(ValueError, match="non-negative"):
            lda.fit(x, y)

    def test_shrinkage_with_svd_raises(self):
        x, y = self._make_separable_data()
        lda = BinaryLDA(solver="svd", shrinkage="auto")
        with pytest.raises(NotImplementedError):
            lda.fit(x, y)
