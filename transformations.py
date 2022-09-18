import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class MeanImputer(BaseEstimator, TransformerMixin):
    """Mean missing value imputer.
    The mean is used to fill NA values of the passed variables.
    The variables need to be numerical."""
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
    """Mode missing value imputer.
    The mode is used to fill NA values of the passed variables.
    The variables need to be categorical or integers."""
    def __init__(self, variables):
        # Check that the variables are of type list
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables

    def fit(self, X, y=None):
        # Learn and persist mode values in a dictionary
        self.imputer_dict_ = X[self.variables].mode().iloc[0].to_dict()
        return self

    def transform(self, X):
        # Note that we copy X to avoid changing the original dataset
        X = X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace=True)
        return X

class CategoryEncoder(BaseEstimator, TransformerMixin):
    """Categorical values of the passed variables
    are encoded with their ratios with respect to the response variable."""
    def __init__(self, features, target):
        # Check that the features are of type list
        if not isinstance(features, list):
            raise ValueError('features should be a list')
        self.features = features
        # Check that the target is a string (column name)
        if not isinstance(target, str):
            raise ValueError('target should be a string (column name)')
        self.target = target       

    def fit(self, X, y=None):
        # Loop over all feature columns (categorical)
        # Create a dictionary which contains the churn ratio
        # associated with each category for each feature column
        encoding_dict = dict()
        for col in self.features:
            col_groups_dict = X.groupby(col).mean()[self.target].to_dict()
            encoding_dict[col] = col_groups_dict
        self.imputer_dict_ = encoding_dict
        return self

    def transform(self, X):
        # Note that we copy X to avoid changing the original dataset
        X = X.copy()
        # Loop over all feature and replace categories by their
        # pre-compted ratio values
        for feature in self.features:
            new_feature = feature + "_" self.target
            X[new_feature] = X[feature]
            X[new_feature].replace(self.imputer_dict_[feature], inplace=True)
        return X


