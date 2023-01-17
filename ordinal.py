import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, loss=None):
        self.base_estimator = base_estimator
        self.loss = loss #callable

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        
        self.classes_ = np.sort(unique_labels(y))
        self.X_ = X
        self.y_ = y
        
        self.n_features_in_ = X.shape[1]
        
        self.estimators_ = {}

        for i in range(self.classes_.shape[0]-1):
            # for each k - 1 ordinal value we fit a binary classification problem
            binary_y = (y > self.classes_[i]).astype(np.uint8)
            clf = clone(self.base_estimator)
            clf.fit(X, binary_y)
            self.estimators_[self.classes_[i]] = clf
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        clfs_predict = {k: self.estimators_[k].predict_proba(X) for k in self.estimators_}
        predicted = []
        
        if len(self.classes_) == 1:
            return np.ones((X.shape[0],1))
        
        # V1 = 1 - Pr(y > V1)
        predicted.append(1 - clfs_predict[self.classes_[0]][:,1])
        
        for i in range(1, len(self.classes_) - 1): #sorted classes
            # Vi = Pr(y > Vi-1) - Pr(y > Vi)
             predicted.append(clfs_predict[self.classes_[i-1]][:,1] - 
                              clfs_predict[self.classes_[i]][:,1])
        
        # Vk = Pr(y > Vk-1)
        predicted.append(clfs_predict[self.classes_[-2]][:,1])
        return np.vstack(predicted).T

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y, sample_weight=None):
        _, indexed_y = np.unique(y, return_inverse=True)
        return self.loss(indexed_y, self.predict(X), sample_weight=sample_weight)