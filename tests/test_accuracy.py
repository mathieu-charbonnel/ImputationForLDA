import numpy as np
import pytest

from src.metrics.accuracy import (
    _compute_feature_mean,
    _compute_feature_variance,
    _majority_vote,
    _weighted_distance,
    acc,
)


class TestWeightedDistance:
    def test_identical_vectors(self):
        x = np.array([1.0, 2.0, 3.0])
        var = np.array([1.0, 1.0, 1.0])
        result = _weighted_distance(x, x, var)
        assert result == 0.0

    def test_with_zero_entries(self):
        x1 = np.array([0.0, 2.0, 3.0])
        x2 = np.array([1.0, 2.0, 3.0])
        var = np.array([1.0, 1.0, 1.0])
        result = _weighted_distance(x1, x2, var)
        assert result == np.sqrt(1.0)

    def test_symmetric(self):
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([4.0, 5.0, 6.0])
        var = np.array([1.0, 2.0, 3.0])
        assert _weighted_distance(x1, x2, var) == _weighted_distance(x2, x1, var)

    def test_non_negative(self):
        x1 = np.array([1.0, -2.0, 3.0])
        x2 = np.array([-1.0, 2.0, -3.0])
        var = np.array([1.0, 1.0, 1.0])
        assert _weighted_distance(x1, x2, var) >= 0


class TestMajorityVote:
    def test_unanimous_ones(self):
        predictions = [
            np.array([1, 1, 1]),
            np.array([1, 1, 1]),
            np.array([1, 1, 1]),
        ]
        result = _majority_vote(predictions)
        np.testing.assert_array_equal(result, np.array([1, 1, 1]))

    def test_unanimous_zeros(self):
        predictions = [
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
        ]
        result = _majority_vote(predictions)
        np.testing.assert_array_equal(result, np.array([0, 0, 0]))

    def test_mixed_votes(self):
        predictions = [
            np.array([1, 0, 1]),
            np.array([1, 0, 0]),
            np.array([0, 1, 1]),
        ]
        result = _majority_vote(predictions)
        np.testing.assert_array_equal(result, np.array([1, 0, 1]))

    def test_tie_goes_to_zero(self):
        predictions = [
            np.array([1, 0]),
            np.array([0, 1]),
        ]
        result = _majority_vote(predictions)
        np.testing.assert_array_equal(result, np.array([0, 0]))


class TestComputeFeatureMean:
    def test_simple_mean(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _compute_feature_mean(data, 2, 2)
        np.testing.assert_array_almost_equal(result, np.array([2.0, 3.0]))

    def test_skips_zeros(self):
        data = np.array([[1.0, 0.0], [3.0, 4.0]])
        result = _compute_feature_mean(data, 2, 2)
        assert result[0] == 2.0
        assert result[1] == 4.0


class TestComputeFeatureVariance:
    def test_zero_variance(self):
        data = np.array([[2.0, 3.0], [2.0, 3.0]])
        mean = np.array([2.0, 3.0])
        result = _compute_feature_variance(data, mean, 2, 2)
        np.testing.assert_array_almost_equal(result, np.array([0.0, 0.0]))

    def test_nonzero_variance(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        mean = np.array([2.0, 3.0])
        result = _compute_feature_variance(data, mean, 2, 2)
        np.testing.assert_array_almost_equal(result, np.array([1.0, 1.0]))


class TestAcc:
    def test_unknown_method_raises(self):
        x = np.array([[1.0, 2.0]])
        y = np.array([0])
        with pytest.raises(ValueError, match="Unknown imputation method"):
            acc(x, y, x, y, "nonexistent_method")

    def test_no_imputation_perfect_data(self):
        np.random.seed(42)
        x_train = np.vstack([
            np.random.randn(50, 2) + np.array([3, 3]),
            np.random.randn(50, 2) + np.array([-3, -3]),
        ])
        y_train = np.array([0] * 50 + [1] * 50)
        x_test = np.vstack([
            np.random.randn(20, 2) + np.array([3, 3]),
            np.random.randn(20, 2) + np.array([-3, -3]),
        ])
        y_test = np.array([0] * 20 + [1] * 20)
        result = acc(x_train, y_train, x_test, y_test, "no_imputation")
        assert 0.0 <= result <= 1.0
        assert result > 0.8

    def test_grand_mean_returns_valid_score(self):
        np.random.seed(42)
        x_train = np.vstack([
            np.random.randn(50, 2) + np.array([3, 3]),
            np.random.randn(50, 2) + np.array([-3, -3]),
        ])
        y_train = np.array([0] * 50 + [1] * 50)
        x_test = np.vstack([
            np.random.randn(20, 2) + np.array([3, 3]),
            np.random.randn(20, 2) + np.array([-3, -3]),
        ])
        y_test = np.array([0] * 20 + [1] * 20)
        result = acc(x_train, y_train, x_test, y_test, "grand_mean")
        assert 0.0 <= result <= 1.0

    def test_regression_returns_valid_score(self):
        np.random.seed(42)
        x_train = np.vstack([
            np.random.randn(50, 3) + np.array([3, 3, 3]),
            np.random.randn(50, 3) + np.array([-3, -3, -3]),
        ])
        y_train = np.array([0] * 50 + [1] * 50)
        x_test = np.vstack([
            np.random.randn(20, 3) + np.array([3, 3, 3]),
            np.random.randn(20, 3) + np.array([-3, -3, -3]),
        ])
        y_test = np.array([0] * 20 + [1] * 20)
        result = acc(x_train, y_train, x_test, y_test, "regression")
        assert 0.0 <= result <= 1.0
