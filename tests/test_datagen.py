import numpy as np
import pytest

from src.loading.datagen import data_generation, full_gen, removing_data


class TestDataGeneration:
    def test_output_lengths(self):
        result = data_generation("normal", 3, 50, 20)
        assert len(result) == 4
        assert len(result[0]) == 50
        assert len(result[1]) == 50
        assert len(result[2]) == 20
        assert len(result[3]) == 20

    def test_feature_dimensionality(self):
        dim = 5
        result = data_generation("normal", dim, 30, 10)
        for sample in result[0]:
            assert len(sample) == dim
        for sample in result[2]:
            assert len(sample) == dim

    def test_labels_are_binary(self):
        result = data_generation("normal", 3, 100, 50)
        for label in result[1]:
            assert label in (0, 1)
        for label in result[3]:
            assert label in (0, 1)

    def test_covariance_type_random(self):
        result = data_generation("random", 3, 20, 10)
        assert len(result[0]) == 20

    def test_covariance_type_str_correlation_higher_index(self):
        result = data_generation("str_correlation_higherIndex", 3, 20, 10)
        assert len(result[0]) == 20

    def test_covariance_type_str_correlation_high_diagonal(self):
        result = data_generation("str_correlation+high_diagonal", 3, 20, 10)
        assert len(result[0]) == 20


class TestRemovingData:
    def _make_data(self, dim=3, n_train=50, n_test=20):
        np.random.seed(42)
        x_train = np.random.randn(n_train, dim) + 5
        y_train = np.array([0] * (n_train // 2) + [1] * (n_train - n_train // 2))
        x_test = np.random.randn(n_test, dim) + 5
        y_test = np.array([0] * (n_test // 2) + [1] * (n_test - n_test // 2))
        return x_train, y_train, x_test, y_test

    def test_mcar_output_structure(self):
        x_train, y_train, x_test, y_test = self._make_data()
        result = removing_data(x_train, y_train, x_test, y_test, "MCAR", 0.3)
        assert len(result) == 6
        assert result[0].shape == x_train.shape
        assert result[2].shape == x_test.shape

    def test_mcar_introduces_zeros(self):
        np.random.seed(0)
        x_train, y_train, x_test, y_test = self._make_data(dim=5, n_train=200)
        result = removing_data(x_train, y_train, x_test, y_test, "MCAR", 0.5)
        assert np.sum(result[0] == 0) > 0

    def test_mcar_first_column_untouched(self):
        np.random.seed(0)
        x_train, y_train, x_test, y_test = self._make_data()
        original_first_col = x_train[:, 0].copy()
        result = removing_data(x_train, y_train, x_test, y_test, "MCAR", 0.5)
        np.testing.assert_array_equal(result[0][:, 0], original_first_col)

    def test_mar_output_structure(self):
        x_train, y_train, x_test, y_test = self._make_data()
        result = removing_data(x_train, y_train, x_test, y_test, "MAR", 0.3)
        assert len(result) == 6

    def test_mnar_output_structure(self):
        x_train, y_train, x_test, y_test = self._make_data()
        result = removing_data(x_train, y_train, x_test, y_test, "MNAR", 0.3)
        assert len(result) == 6

    def test_intact_data_preserved(self):
        x_train, y_train, x_test, y_test = self._make_data()
        x_train_orig = x_train.copy()
        x_test_orig = x_test.copy()
        result = removing_data(x_train, y_train, x_test, y_test, "MCAR", 0.3)
        np.testing.assert_array_equal(result[4], x_train_orig)
        np.testing.assert_array_equal(result[5], x_test_orig)


class TestFullGen:
    def test_output_structure(self):
        np.random.seed(42)
        result = full_gen("normal", 3, 50, 20, "MCAR", 0.2)
        assert len(result) == 6

    def test_output_shapes(self):
        np.random.seed(42)
        result = full_gen("normal", 4, 30, 15, "MCAR", 0.2)
        assert result[0].shape[1] == 4 or len(result[0][0]) == 4
