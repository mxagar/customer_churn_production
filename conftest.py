'''Testing configuration module for Pytest.
This file is read by pytest and the fixtures
defined in it are used in all the tested files.

Author: Mikel Sagardia
Date: 2022-09-20
'''

import pytest

# Source of tested functions/classes
import churn_library as cl

# Fixtures of the churn library functions.
# Fixtures are predefined variables passed to test functions;
# in this case, most variables are functions/classes to be tested.

@pytest.fixture
def dataset_path_train():
    '''Dataset path for training'''
    return "./data/bank_data.csv"

@pytest.fixture
def dataset_path_inference():
    '''Dataset path for inference'''
    return "./data/bank_data_sample.csv"

@pytest.fixture
def import_data():
    '''import_data function from churn_library'''
    return cl.import_data

@pytest.fixture
def eda_path():
    '''Path where EDA images are saved'''
    return './images/eda'

@pytest.fixture
def artifact_path():
    '''Path where the processing artifacts images are saved'''
    return './artifacts'

@pytest.fixture
def expected_artifact():
    '''Local name of the saved artifact'''
    return 'processing_params.pkl'

@pytest.fixture
def expected_eda_images():
    '''List of saved EDA image filenames'''
    return ['corr_heatmap.png',
            'total_trans_ct_dist.png',
            'age_dist.png',
            'churn_dist.png',
            'marital_status_dist.png']

@pytest.fixture
def perform_eda():
    '''perform_eda function from churn_library'''
    return cl.perform_eda

@pytest.fixture
def category_lst():
    '''List of categorical features'''
    return ['Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category']

@pytest.fixture
def response():
    '''Response/Target variable name'''
    return "Churn"

@pytest.fixture
def num_features():
    '''Number of final features'''
    return 19

@pytest.fixture
def perform_data_processing():
    '''perform_data_processing function from churn_library'''
    return cl.perform_data_processing

@pytest.fixture
def models_path():
    '''Path where models are stored'''
    return './models'

@pytest.fixture
def results_path():
    '''Path where result images are stored'''
    return './images/results'

@pytest.fixture
def expected_models():
    '''List of stored model names'''
    return ['logistic_regression_model_pipe.pkl',
            'random_forest_model_pipe.pkl']

@pytest.fixture
def expected_result_images():
    '''List of saved result images'''
    return ['rf_classification_report.png',
            'lr_classification_report.png',
            'feature_importance_random_forest.png',
            'roc_plots.png']

@pytest.fixture
def train_models():
    '''train_models function from churn_library'''
    return cl.train_models

@pytest.fixture
def evaluate_models():
    '''evaluate_models function from churn_library'''
    return cl.evaluate_models

@pytest.fixture
def load_model_pipeline():
    '''load_model_pipeline function from churn_library'''
    return cl.load_model_pipeline

@pytest.fixture
def predict():
    '''predict function from churn_library'''
    return cl.predict

@pytest.fixture
def split():
    '''split function from churn_library'''
    return cl.split

def df_plugin():
    '''Initialize pytest dataset container df as None'''
    return None

def splits_plugin():
    '''Initialize pytest splits container as None
    splits = (X_train, X_test, y_train, y_test)
    '''
    return None

def models_plugin():
    '''Initialize pytest model container as None
    models = (lr, rf)
    '''
    return None

def pytest_configure():
    '''Create objects in namespace:
    - `pytest.df`
    - `pytest.splits`
    - `pytest.model`
    '''
    pytest.df = df_plugin() # we can access & modify pytest.df in test functions!
    pytest.splits = splits_plugin()
    pytest.models = models_plugin()
