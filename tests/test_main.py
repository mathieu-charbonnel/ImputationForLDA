import argparse

import pytest

from main import validate_probs_missingness


class TestValidateProbsMissingness:
    def test_valid_zero(self):
        assert validate_probs_missingness("0") == 0.0

    def test_valid_one(self):
        assert validate_probs_missingness("1") == 1.0

    def test_valid_middle(self):
        assert validate_probs_missingness("0.5") == 0.5

    def test_invalid_negative(self):
        with pytest.raises(argparse.ArgumentTypeError):
            validate_probs_missingness("-0.1")

    def test_invalid_above_one(self):
        with pytest.raises(argparse.ArgumentTypeError):
            validate_probs_missingness("1.5")

    def test_invalid_non_numeric(self):
        with pytest.raises(ValueError):
            validate_probs_missingness("abc")
