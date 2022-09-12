import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y


class CCBPClassifier(BaseEstimator, ClassifierMixin):
    """
    Binary Classifier implementing the Continuous correlated beta process model.

    Read more in  Goetschalckx, R., Poupart, P., and Hoey, J. Continuous correlated beta processes. In IJCAI, 2011.

    Parameters
    ----------
    lengths : {int, float, or nd.array of shape (num_features)}, default = 0.2
        The length hyperparamter for the Continuous correlated beta process model.
        Specifies the distance along a numeric feature dimension for which data should be considered 'similar'
        Increasing this value increases the change allowed along a feature dimension whilst that is required for data
        to be reduced to a certain level of similarity.

    priors : list of length 2, default [0.5, 0.5]
        The prior alpha and beta values for the Continuous correlated beta process model.

        Either [0.5, 0.5] or [1, 1] may be reasonable.
        See https://en.wikipedia.org/wiki/Beta_distribution#Rule_of_succession for discussion on this.

        If experts have prior belief about the average probability of success
        that should be expected across samples p, then set equal to
            [0.5 + p*c, 0.5 + (1-p)*c] or [1 + p*c, 1 + (1-p)*c]
        where c is the 'confidence' in this prior. I would suggest using c<3, as large c
        may prevent data from effecting the model predictions.

        Call
            instance_of_CCBP_model.set_priors(probability, confidence, jeffery=True/False)
        to do this automatically.

    return_evidence_level : bool, default = False
        Determines whether predict and predict_proba should return a confidence metric.

        If True, predicting will return both the standard predictions and an array
        of shape (n_queries) representing the amount of evidence than the model believes
        supports that prediction.

        If False, predictions will return as normal for an estimator.

     Examples
    --------
    >> X = np.asarray([[1,2,3], [4,5,6], [7,8,9], [10,11,12]]) # of shape (n_samples, n_features)
    >> y = np.asarray([0, 0, 1, 1]) #  of shape (n_samples,)

    >> CCBP = CCBPClassifier(lengths=np.array([1,2,4]))
    >> CCBP.fit(X, y)

    >> CCBP.predict(np.asarray([[13,14,15],[1,1,1]])))
    [1 0]
    >> CCBP.predict_proba(np.asarray([[13,14,15],[1,1,1]]))
    [[0.49999996 0.50000004]
    [0.55018378 0.44981622]]

    >> CCBP.set_params(**{"return_evidence_level":True})
    >> prediction, evidence_level = CCBP.predict(np.asarray([[13,14,15],[1,1,1]])))
    >> prediction
    [1, 0]
    >> evidence_level
    [2.00000014, 2.22313016]

    >> CCBP.set_priors(0.7, 2, jeffery=False)
    >> prediction, evidence_level = CCBP.predict(np.asarray([[13,14,15],[1,1,1]])))
    >> prediction
    [1, 1]
    >> evidence_level
    [4.00000014, 4.22313016]
    """

    def __init__(
            self,
            lengths=0.2,
            priors=[1.0, 1.0],
            return_evidence_level=False
    ):
        assert type(lengths) == np.ndarray or type(lengths) == int or type(lengths) == float
        self.lengths = lengths
        self.priors = priors
        self.return_evidence_level = return_evidence_level

    def set_priors(self, probability, confidence=3, jeffery=True):
        """
        Sets up the priors parameters using a pre-known probability and confidence level.

        Parameters
        ----------
        jeffery : bool, default = True
            jeffery = True uses the formula
                [0.5 + p*c, 0.5 + (1-p)*c]
            jeffery = False uses the formula
                [1.0 + p*c, 1.0 + (1-p)*c]
        See https://en.wikipedia.org/wiki/Beta_distribution#Rule_of_succession for discussion on this.

        Returns
        -------
        self : CCBPClassifier
                The Continuous correlated beta process classifier.



        """
        if jeffery:
            self.priors = [0.5 + probability * confidence, 0.5 + (1 - probability) * confidence]
        else:
            self.priors = [1.0 + probability * confidence, 1.0 + (1 - probability) * confidence]

    def fit(self, X, y):

        """
        Fit the k-nearest neighbors classifier from the training dataset.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
            (n_samples, n_samples) if metric='precomputed'
            Training data.
        y : {array-like, sparse matrix of bools} of shape (n_samples,) Target values.


        Returns
        -------
        self : CCBPClassifier
                The fitted Continuous correlated beta process classifier.
        """

        X, y = check_X_y(X, y)

        self._fit_X = X

        self.success_vec = np.where(y == 1, 1, 0)
        self.failure_vec = 1 - self.success_vec

        return self

    def kernel(self, X):
        """
        Calculates similarity matrix and resultant predictions for query data
        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Test samples.
        Returns
        -------
        predicted_probabilities : ndarray of shape (n_queries,)
            Probability of success for each  of the test samples.

        evidence_level (optional) : ndarray of shape (n_queries,)
            If self.return_evidence_level is True then an array of evidence_levels is also returned
        """
        # Ensures correct matrix shapes
        if type(self.lengths) == np.ndarray:
            assert np.size(self.lengths) == np.shape(X)[1] or np.size(self.lengths) == 1
        assert np.shape(X)[1] == np.shape(self._fit_X)[1]

        # Reshape for quick matrix operations
        _X = np.repeat(X[:, np.newaxis, :], np.shape(self._fit_X)[0], axis=1)
        _fit_X = np.repeat(self._fit_X[np.newaxis, :, :], np.shape(X)[0], axis=0)

        if type(self.lengths) == int or type(self.lengths) == float:
            _lengths = np.ones_like(_X) * self.lengths
        else:
            _lengths = np.repeat(self.lengths[np.newaxis, :], np.shape(X)[0], axis=0)
            _lengths = np.repeat(_lengths[:, np.newaxis, :], np.shape(self._fit_X)[0], axis=1)

        #
        matrix = (_X - _fit_X)  # shape is (n_samples_X_to_predict, n_samples_X_data, n_features)
        matrix = np.square(matrix)  # shape is (n_samples_X_to_predict, n_samples_X_data, n_features)
        matrix = matrix / _lengths  # shape is (n_samples_X_to_predict, n_samples_X_data, n_features)
        matrix = np.sum(matrix, axis=2).astype(float)  # shape is (n_samples_X_to_predict, n_samples_X_data)
        matrix = np.exp(-matrix)  # shape is (n_samples_X_to_predict, n_samples_X_data)

        alphas = np.sum(matrix * self.success_vec, axis=1) + self.priors[0]  # shape is (n_samples_X_to_predict)
        betas = np.sum(matrix * self.failure_vec, axis=1) + self.priors[1]  # shape is (n_samples_X_to_predict)

        predicted_probabilities = alphas / (alphas + betas)  # mean

        if self.return_evidence_level:
            evidence_level = alphas + betas
            return predicted_probabilities, evidence_level
        return predicted_probabilities

    def predict(self, X):
        """
        Predict the boolean class labels for the provided data.
        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
                or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.
        Returns
        -------
        y : ndarray of shape (n_queries,) or (n_queries, n_outputs)
            Class labels for each data sample.

        evidence_level (optional) : ndarray of shape (n_queries,)
            If self.return_evidence_level is True then an array of evidence_levels is also returned
        """
        prediction = self.kernel(X)

        if self.return_evidence_level:
            prediction, evidence_level = prediction
            return np.where(prediction >= 0.5, 1, 0), evidence_level
        return np.where(prediction >= 0.5, 1, 0)

    def predict_proba(self, X):

        """
        Return probability estimates for the test data X.
        Parameters
        ----------
        X : array-like of shape (n_queries, n_features), \
            or (n_queries, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        p : ndarray of shape (n_queries, n_classes), or a list of n_outputs \
            of such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered [0, 1].

        evidence_level (optional) : ndarray of shape (n_queries,)
            If self.return_evidence_level is True then an array of evidence_levels is also returned
        """
        prediction = self.kernel(X)

        if self.return_evidence_level:
            prediction, evidence_level = prediction
            return np.column_stack((1 - prediction, prediction)), evidence_level
        return np.column_stack((1 - prediction, prediction))
