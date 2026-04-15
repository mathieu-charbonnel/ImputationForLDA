import numpy as np
import pytest

from src.models.binary_lda import BinaryLDA


class TestBinaryLDAClassification:
    def _make_fitted_lda(self):
        np.random.seed(42)
        x = np.vstack([
            np.random.randn(50, 2) + np.array([3, 3]),
            np.random.randn(50, 2) + np.array([-3, -3]),
        ])
        y = np.array([0] * 50 + [1] * 50)
        lda = BinaryLDA(solver="eigen")
        lda.fit(x, y)
        return lda

    def test_decision_function_shape(self):
        lda = self._make_fitted_lda()
        x0 = np.array([[1.0, 2.0], [3.0, 4.0]])
        x1 = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = lda.decision_function(x0, x1)
        assert result.shape == (2,)

    def test_decision_function_wrong_features_raises(self):
        lda = self._make_fitted_lda()
        x0 = np.array([[1.0, 2.0, 3.0]])
        x1 = np.array([[5.0, 6.0]])
        with pytest.raises(ValueError, match="features per sample"):
            lda.decision_function(x0, x1)

    def test_predict_returns_classes(self):
        lda = self._make_fitted_lda()
        x0 = np.array([[10.0, 0.0], [0.0, 0.0]])
        x1 = np.array([[0.0, 0.0], [0.0, 10.0]])
        result = lda.predict(x0, x1)
        assert all(r in [0, 1] for r in result)

    def test_score_returns_float(self):
        lda = self._make_fitted_lda()
        x0 = np.array([[1.0, 0.0]])
        x1 = np.array([[0.0, 1.0]])
        y = np.array([1])
        result = lda.score(x0, x1, y)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
