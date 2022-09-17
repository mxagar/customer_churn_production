import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class MeanImputer(BaseEstimator, TransformerMixin):
    """Numerical missing value imputer.
    The mean is used to fill NA values of the passed variables."""

    def __init__(self, variables):
        # Check that the variables are of type list
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables

    def fit(self, X, y=None):
        # Learn and persist mean values in a dictionary
        self.imputer_dict_ = X[self.variables].mean().to_dict()
        return self

    def transform(self, X):
        # Note that we copy X to avoid changing the original dataset
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X
    
class ModeImputer(BaseEstimator, TransformerMixin):
    """Categorical missing value imputer.
    The mode is used to fill NA values of the passed variables."""
    pass

class CategoryEncoder(BaseEstimator, TransformerMixin):
    """Categorical values of the passed variables
    are encoded with their ratios with respect to the response variable."""
    pass


