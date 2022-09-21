'''This module tests the functions in the module churn_library.py.
That tested module  defined the necessary data science processing steps
to build and save models that predict customer churn using the
Credit Card Customers dataset from Kaggle:

https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code

Altogether, 7 unit tests are defined using pytest:
- test_run_setup(run_setup)
- test_import(import_data)
- test_eda(perform_eda)
- test_perform_data_processing(perform_data_processing)
- test_train_models(train_models)
- test_evaluate_models(evaluate_models)
- test_predict(predict)

Clean code principles are guaranteed in the project
- Modularized code
- PEP8 conventions
- Error handling
- Testing
- Logging

PEP8 conventions checked with:

>> pylint test_churn_library.py.py # 8.48/10
>> autopep8 test_churn_library.py.py

Since the we use the logging module in the tests,
the testing file must be called explicitly,
not with `pytest` alone:

>> python test_churn_library.py

Note that the "__main__" calls pytest.
Additionally, note that the testing configuration fixtures
are located in `conftest.py`.

The content from `conftest.py` must be consistent with the
project configuration file `config.yaml`.

To install pytest:

>> pip install -U pytest

The script expects the proper dataset to be located in `./data`

Additionally:

- Any produced models are stored in `./models`
- Any plots of the EDA and the classification results are stored in `./images/`
- Logs are stored in `./logs/`
- All other artifacts (e.g., data processing parameters) are stored in `./artifacts/`

Author: Mikel Sagardia
Date: 2022-06-08
'''

import os
from os import listdir
from os.path import isfile, join
import logging
import joblib
# Without logging and with fixtures in conftest.py
# we'd need to import pytest only in conftest.py
import pytest
import numpy as np

#from customer_churn.transformations import MeanImputer, ModeImputer, CategoryEncoder
#import customer_churn.transformations as tf

# Logging configuration
logging.basicConfig(
    filename='./logs/test_churn_library.log', # filename, where it's dumped
    level=logging.INFO, # minimum level I log
    filemode='w',
    # https://docs.python.org/3/library/logging.html
    # logger - time - level - our message
    format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')

# IMPORTANT: the file conftest.py defines the fixtures used in here!

### -- Tests -- ###

def test_run_setup(config_filename, run_setup):
    '''Test project setup function.

    Input:
        config_filename (function object): fixture function which returns the path
            of the configuration file
        run_setup (function object): fixture function which returns the function to test
    Output:
        None
    '''
    pytest.config_dict = run_setup(config_filename=config_filename)
    
    # Check folders
    # data_path: ./data
    # eda_output_path: ./images/eda
    # artifact_path: ./artifacts
    # model_output_path: ./models
    # eval_output_path: ./images/results
    # ./logs
    folders = [pytest.config_dict["data_path"],
               pytest.config_dict["eda_output_path"],
               pytest.config_dict["artifact_path"],
               pytest.config_dict["model_output_path"],
               pytest.config_dict["eval_output_path"],
               "./logs"]
    for folder in folders:
        try:
            assert os.path.exists(folder)
        except AssertionError as err:
            logging.error("TESTING run_setup: ERROR - Necessary folders are not in place: %s", folder)

def test_import_data(import_data, dataset_path_train, dataset_path_inference):
    '''Test data import.

    Input:
        import_data (function object): function to be tested
        dataset_path_train (function object): fixture function which returns the path
            of the dataset to be imported for training
        dataset_path_inference (function object): fixture function which returns the path
            of the dataset to be imported for inference
    Output:
        None
    '''
    # Training dataset
    try:
        df = import_data(dataset_path_train, save_sample=True)
        # Assign to pytest namespace object for further use
        pytest.df = df
        logging.info("TESTING import_data (train): SUCCESS")
    except FileNotFoundError as err:
        logging.error("TESTING import_data (train): ERROR - The file wasn't found")
        raise err
    try:
        assert pytest.df.shape[0] > 0
        assert pytest.df.shape[1] > 0
    except AssertionError as err:
        logging.error("TESTING import_data (train): ERROR - File has no rows / columns")
        raise err

    # Inference dataset
    try:
        df_sample = import_data(dataset_path_inference, save_sample=False)
        logging.info("TESTING import_data (inference): SUCCESS")
        try:
            assert df_sample.shape[0] > 0
            assert df_sample.shape[1] > 0
        except AssertionError as err:
            logging.error("TESTING import_data (inference): ERROR - File has no rows / columns")
            raise err
    except FileNotFoundError as err:
        logging.error("TESTING import_datas (inference): ERROR - The file wasn't found")
        raise err

def test_perform_eda(perform_eda, eda_path, expected_eda_images):
    '''Test perform_eda function.

    Input:
        perform_eda (function object): function to be tested
        eda_path (function object): fixture function which returns the path
            where the EDA images are to be saved
        expected_eda_images (function object): fixture function which returns a
            list of all the expected EDA image filenames
    Output:
        None
    '''
    # After perform_eda(df) we should get these images:
    perform_eda(pytest.df, eda_path)
    filenames = [f for f in listdir(eda_path) if isfile(join(eda_path, f))]

    try:
        assert len(filenames) >= len(expected_eda_images)
    except AssertionError as err:
        logging.error("TESTING perform_eda: ERROR - Missing EDA images")
        raise err

    for image in expected_eda_images:
        try:
            assert image in filenames
        except AssertionError as err:
            logging.error("TESTING perform_eda: ERROR - The image %s is missing", image)
            raise err

    logging.info("TESTING perform_eda: SUCCESS")

def test_perform_data_processing(perform_data_processing,
                                 split,
                                 num_features,
                                 artifact_path,
                                 response,
                                 expected_artifact):
    '''Test perform_data_processing function.

    Input:
        perform_data_processing (function object): function to be tested
        split (function object): split dataset
        num_features (function object): fixture function which returns the number
            of features in the final dataset
        artifact_path (function object): fixture which returns the path
            of the artifacts related to the data processing
        response (function object): fixture function which returns the name
            of the target/response
    Output:
        None
    '''
    # Data Processing: Data Cleaning, Feature Engineering
    X, y = perform_data_processing(pytest.df,
                                   response=response,
                                   artifact_path=artifact_path,
                                   train=True)
    X_train, X_test, y_train, y_test = split(X, y)
    # Save training splits
    pytest.splits = (X_train, X_test, y_train, y_test)
    try:
        assert X_train.shape[0] > 0
        assert X_train.shape[1] == num_features
        assert X_test.shape[0] > 0
        assert X_test.shape[1] == X_train.shape[1]
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
    except AssertionError as err:
        logging.error("TESTING perform_data_processing: ERROR - Unexpected sizes for X & y!")
        raise err

    # Check the features:
    # - no duplicates
    # - no missing values
    # - all are numerical
    cols_num = list(X.select_dtypes(['int64','float64']).columns)
    cols_cat = list(X.select_dtypes(['object']).columns)
    try:
        assert X.index.is_unique
    except AssertionError as err:
        logging.error("TESTING perform_data_processing: ERROR - Duplicate indices in processed dataset!")
        raise err
    try:
        assert int(np.sum(X.isnull().sum()==0)) == int(X.shape[1])
    except AssertionError as err:
        logging.error("TESTING perform_data_processing: ERROR - Missing values in the processed dataset!")
        raise err
    try:
        assert len(cols_cat) == 0
    except AssertionError as err:
        logging.error("TESTING perform_data_processing: ERROR - There are categorical columns in the processed dataset!")
        raise err
    try:
        assert len(cols_num) == X.shape[1]
    except AssertionError as err:
        logging.error("TESTING perform_data_processing: ERROR - Not all columns in the processed dataset are numerical!")
        raise err

    # Load processing parameters
    processing_params = dict()
    try:
        processing_params = joblib.load(join(artifact_path, expected_artifact))
    except Exception as err:
        logging.error("TESTING perform_data_processing: ERROR - Processing parameters dictionary %s cannot be loaded", expected_artifact)
        raise err

    # Check the data processing artifact
    # - cols_cat
    # - cols_num
    # - num_features == 19
    # - mean_imputer: correctly filled dictionary
    # - mode_imputer: correctly filled dictionary
    # - category_encoder: correctly filled dictionary
    try:
        assert processing_params["num_features"] == num_features
    except AssertionError as err:
        logging.error("TESTING perform_data_processing: ERROR - Number of features in training is different to expected: %d != %d.",
                      processing_params["num_features"], num_features)
        raise err
    for feature in processing_params["mean_imputer"].variables:
        try:
            assert feature in processing_params["mean_imputer"].imputer_dict_.keys()
        except AssertionError as err:
            logging.error("TESTING perform_data_processing: ERROR - MeanImputer is inconsistent.")
            raise err
    for feature in processing_params["mode_imputer"].variables:
        try:
            assert feature in processing_params["mode_imputer"].imputer_dict_.keys()
        except AssertionError as err:
            logging.error("TESTING perform_data_processing: ERROR - ModeImputer is inconsistent.")
            raise err
    for _, categories in processing_params["category_encoder"].imputer_dict_.items():
        for _, ratio in categories.items():
            try:
                assert ratio >= 0.0
                assert ratio <= 1.0
            except AssertionError as err:
                logging.error("TESTING perform_data_processing: ERROR - CategoryEncoder contains values out of range.")
                raise err

    logging.info("TESTING perform_data_processing: SUCCESS")

def test_train_models(train_models,
                      load_model_pipeline,
                      models_path,
                      expected_models):
    '''
    Test train_models

    Input:
        train_models (function object): function to be tested
        load_model_pipeline (function object): auxiliary function that loads models,
            also tested
        models_path (function object): fixture function which returns the path
            where the created models are to be found
        expected_models (function object): fixture function which returns
            the names of the models that should have been stored by train_models
    Output:
        None
    '''
    # Unpack training splits stored in the pytest namespace
    X_train, X_test, y_train, y_test = pytest.splits

    # Perform training
    models = train_models(X_train, y_train, model_output_path=models_path)
    #models = (lr_pipe_cv_best, rf_pipe_cv_best)
    pytest.models = models

    # Check that models are correctly stored
    model_filenames = [f for f in listdir(models_path) if isfile(join(models_path, f))]
    try:
        assert len(model_filenames) >= len(expected_models)
    except AssertionError as err:
        logging.error("TESTING train_models: ERROR - Missing models")
        raise err

    for model in expected_models:
        try:
            assert model in model_filenames
        except AssertionError as err:
            logging.error("TESTING train_models: ERROR - The model %s is missing", model)
            raise err

    # Check that the models can be loaded
    for model_name in model_filenames:
        if model_name in expected_models:
            try:
                #model = joblib.load(join(models_path, model_name))
                model = load_model_pipeline(join(models_path, model_name))
            except Exception as err:
                logging.error("TESTING train_models: ERROR - Model %s cannot be loaded", model_name)
                raise err

    logging.info("TESTING train_models: SUCCESS")

def test_evaluate_models(evaluate_models,
                         results_path,
                         expected_result_images):
    '''
    Test evaluate_models function.

    Input:
        evaluate_models (function object): function to be tested
        results_path (function object): fixture function which returns the path
            where the created result images are to be found
        expected_result_images (function object): fixture function which returns
            the names of the result images that should have been stored
            by train_models
    Output:
        None
    '''
    # Unpack the models stored in the pytest namespace
    models = pytest.models
    # Unpack training splits stored in the pytest namespace
    X_train, X_test, y_train, y_test = pytest.splits

    # Call evaluate_models
    evaluate_models(X_train,
                    X_test,
                    y_train,
                    y_test,
                    models,
                    eval_output_path=results_path)

    # Check that the result images were correctly saved
    result_filenames = [f for f in listdir(results_path) if isfile(join(results_path, f))]
    try:
        assert len(result_filenames) >= len(expected_result_images)
    except AssertionError as err:
        logging.error("TESTING evaluate_models: ERROR - Missing models")
        raise err

    for result in result_filenames:
        try:
            assert result in result_filenames
        except AssertionError as err:
            logging.error("TESTING evaluate_models: ERROR - The result image %s is missing", result)
            raise err

    logging.info("TESTING evaluate_models: SUCCESS")

def test_predict(predict):
    '''
    Test predict function.

    Input:
        predict (function object): function to be tested
    Output:
        None
    '''
    # Unpack the models stored in the pytest namespace
    models = pytest.models
    # Unpack training splits stored in the pytest namespace
    X_train, X_test, y_train, y_test = pytest.splits

    for model in models:
        # Predict
        preds =  predict(model, X_test)

        try:
            assert len(preds) == len(y_test)
        except AssertionError as err:
            logging.error("TESTING predict: ERROR - Wrong number of predictions")
            raise err

        accuracy = float(np.sum(preds == y_test)) / float(len(y_test))
        try:
            assert accuracy > 0.6
        except AssertionError as err:
            logging.error("TESTING predict: ERROR - Accuracy lower than 0.6: %f.", accuracy)
            raise err

    logging.info("TESTING predict: SUCCESS")

if __name__ == "__main__":
    # Without logging, we would run
    # >> pytest
    # or, if the testing file does not start with test_*
    # >> pytest file.py
    # However, logging does not occur when invoking pytest that way.
    # If we want to have logging with pytest, we either configure the TOML / INI
    # or we define the line below in __main__ and execute the tests with
    # >> python test_churn_library.py
    # Sources:
    # https://stackoverflow.com/questions/4673373/logging-within-pytest-tests
    # https://stackoverflow.com/questions/31793540/how-to-save-pytests-results-logs-to-a-file
    pytest.main(args=['-s', os.path.abspath(__file__)])
