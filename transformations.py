'''This module contains the necessary transformation classes
used in the function perform_data_processing()
from the module churn_library.py

Author: Mikel Sagardia
Date: 2022-09-19
'''
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
        self.imputer_dict_ = dict()

    def fit(self, X, y=None):
        '''Learn and persist mean values in a dictionary.'''
        self.imputer_dict_ = X[self.variables].mean().to_dict()
        return self

    def transform(self, X):
        '''Replace the NA values with the learned mean values.'''
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
        self.imputer_dict_ = dict()

    def fit(self, X, y=None):
        '''Learn and persist mean values in a dictionary.'''
        self.imputer_dict_ = X[self.variables].mode().iloc[0].to_dict()
        return self

    def transform(self, X):
        '''Replace the NA values with the learned mode values.'''
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
        self.imputer_dict_  = dict()
        self.encoded_categoricals_ = []

    def fit(self, X, y=None):
        '''Learn and persist group ratios in a dictionary.'''
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
        '''Replace the NA values with the learned group ratios.'''
        # Note that we copy X to avoid changing the original dataset
        X = X.copy()
        # Loop over all feature and replace categories by their
        # pre-compted ratio values
        #self.encoded_categoricals_ = []
        for feature in self.features:
            new_feature = feature + "_" + self.target
            self.encoded_categoricals_.append(new_feature)
            X[new_feature] = X[feature]
            X[new_feature].replace(self.imputer_dict_[feature], inplace=True)
        return X
