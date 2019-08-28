"""
Generalized Linear models.
"""
from abc import ABCMeta, abstractmethod
import numbers
import warnings

import numpy as np
import scipy.sparse as sp
from scipy import linalg
from scipy import sparse
from scipy.special import expit
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.sparsefuncs import mean_variance_axis, inplace_column_scale
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

class LinearClassifierMixin():
    """Mixin for linear classifiers.
    Handles prediction for sparse and dense X.
    """
    _estimator_type = "classifier"

    def score(self, X0,X1, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X0,X1), sample_weight=sample_weight)


    def decision_function(self, X0,X1):
        """Predict confidence scores for samples.
        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.
        """
        #if not hasattr(self, 'coef_') or self.coef_ is None:
        #    raise NotFittedError("This %(name)s instance is not fitted "
        #                         "yet" % {'name': type(self).__name__})

        #X0 = check_array(X0, accept_sparse='csr')
        #X1 = check_array(X1, accept_sparse='csr')

        n_features = self.coef_.shape[1]
        if X0.shape[1] != n_features:
            raise ValueError("X0 has %d features per sample; expecting %d"
                             % (X0.shape[1], n_features))
        coef0=self.coef_[0,:]
        coef1=self.coef_[1,:]
        intercept0=self.intercept_[0]
        intercept1=self.intercept_[1]

        score0 = safe_sparse_dot(X0, coef0.T,
                                 dense_output=True) + intercept0
        score1 = safe_sparse_dot(X1, coef1.T,
                                 dense_output=True) + intercept1
        #for i in range(len(scores0)):
        #    scores0[i][1]=scores1[i][0]
        return (score1-score0)

        #return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X0,X1):
        """Predict class labels for samples in X.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """
        scores = self.decision_function(X0,X1)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]

    def _predict_proba_lr(self, X):
        """Probability estimation for OvR logistic regression.
        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        prob = self.decision_function(X)
        expit(prob, out=prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob
