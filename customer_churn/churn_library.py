'''This module performs the necessary data science processing steps
to build and save models that predict customer churn using the
Credit Card Customers dataset from Kaggle:

https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code

The file is divided into two functions contained in main() that run two
typical pipelines which share some steps:
1. Model generation pipeline, with these steps:
    - The dataset is loaded
    - Exploratory Data Analysis (EDA)
    - Data Processing: Cleaning and Feature Engineering (FE)
    - Train/Test split
    - Training: Random Forest and Logistic Regression models are fit
    - Classification report plots
2. Exemplary inference pipeline, with these steps:
    - Exemplary dataset is loaded (generated in pipeline 1)
    - Model pipeline is loaded (generated in pipeline 1)
    - Data Processing: Cleaning and Feature Engineering (FE) (same as in pipeline 1)
    - Prediction

Note that the first pipeline needs to have been executed
before executing the second one; however, once the first has been executed
we can execute only the second one (i.e., we can comment out the first).

Clean code principles are guaranteed:
- Modularized code
- PEP8 conventions
- Error handling
- Testing is carried in the companion file: test_churn_library.py

PEP8 conventions checked with:

>> pylint churn_library.py # 7.96/10
>> autopep8 churn_library.py

The file can be run stand-alone:

>> python churn_library.py

The script expects the proper dataset to be located in `./data`

Additionally:

- The produced models are stored in `./models`
- The plots of the EDA and the classification results are stored in `./images`
- All other artifacts are stored in `./artifacts`

Author: Mikel Sagardia
Date: 2022-06-08
'''

import os
import time
#os.environ['QT_QPA_PLATFORM']='offscreen'
import logging
import joblib
import yaml

#import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # It needs to be called here, otherwise we get error!
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

from transformations import MeanImputer, ModeImputer, CategoryEncoder

# Set library/module options
os.environ['QT_QPA_PLATFORM']='offscreen'
matplotlib.use('TkAgg')
sns.set()

# Logging configuration
logging.basicConfig(
    filename='./logs/churn_library.log', # filename, where it's dumped
    level=logging.INFO, # minimum level I log
    filemode='w',
    # https://docs.python.org/3/library/logging.html
    # logger - time - level - our message
    format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')

def import_data(pth, save_sample=True):
    '''Returns dataframe for the CSV found at pth.
    For training save_sample=True; for inference save_sample=False.
    The previously saved data sample contains some entries of the training dataset
    and it can be used to test the inference.

    Input:
            pth (str): a path to the csv
            save_sample (bool): whether a small testing sample needs to be saved
    Output:
            df (pandas.DataFrame): pandas dataframe with the dataset
    '''
    try:
        # Read file
        data = pd.read_csv(pth)
        # Save sample for testing inference
        if save_sample:
            pth_sample = pth.split('.csv')[0]+'_sample.csv'
            data_sample = data.iloc[:10,:]
            data_sample.to_csv(pth_sample,sep=',', header=True, index=False)
        logging.info("import_data: SUCCESS!")
        return data
    except (FileNotFoundError, NameError):
        #print("File not found!")
        logging.error("import_data: File not found: %s.", pth)
    except IsADirectoryError:
        #print("File is a directory!")
        logging.error("import_data: File is a directory.")

    return None

def perform_eda(data, output_path):
    '''Performs EDA on df and saves figures to the output_path folder.

    Input:
            df (pandas.DataFrame): dataset
            output_path (string): path where report images are saved
    Output:
            None
    '''

    # Check that all columns to be analyzed are present in the dataset
    cols_analyze = ['Attrition_Flag', 'Customer_Age', 'Marital_Status', 'Total_Trans_Ct']
    try:
        for col in cols_analyze:
            assert col in data.columns
        logging.info("perform_eda: All columns to be analyzed are in the dataset.")
    except AssertionError as err:
        #print("EDA: Missing columns in the dataset.")
        logging.error("perform_eda: Missing columns in the dataset.")
        raise err

    # General paramaters
    figsize = (20,15)
    dpi = 200
    rootpath = output_path # './images/eda'

    # New Churn variable: 1 Yes, 0 No
    data['Churn'] = data['Attrition_Flag'].apply(lambda val:
                                                 0 if val == "Existing Customer" else 1)
    # Figure 1: Churn distribution (ratio)
    fig = plt.figure(figsize=figsize)
    data['Churn'].hist()
    fig.savefig(rootpath+'/churn_dist.png', dpi=dpi)

    # Figure 2: Age distribution
    fig = plt.figure(figsize=figsize)
    data['Customer_Age'].hist()
    fig.savefig(rootpath+'/age_dist.png', dpi=dpi)

    # Figure 3: Marital status distribution
    fig = plt.figure(figsize=figsize)
    data['Marital_Status'].value_counts('normalize').plot(kind='bar')
    fig.savefig(rootpath+'/marital_status_dist.png', dpi=dpi)

    # Figure 4: Total transaction count distribution
    fig = plt.figure(figsize=figsize)
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve
    # obtained using a kernel density estimate
    sns.histplot(data['Total_Trans_Ct'], stat='density', kde=True)
    fig.savefig(rootpath+'/total_trans_ct_dist.png', dpi=dpi)

    # Figure 5: Correlations
    fig = plt.figure(figsize=figsize)
    sns.heatmap(data.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    fig.savefig(rootpath+'/corr_heatmap.png', dpi=dpi)

    logging.info("perform_eda: SUCCESS!")

def perform_data_processing(data,
                            response="Churn",
                            artifact_path="./artifacts",
                            train=True):
    '''Performs basic Data Processing and Feature Engineering:
    - basic cleaning,
    - select/drop features,
    - encode categoricals,
    - data validation.

    This function is called for generating the model and for the inference,
    because for both cases, the same data preparation is required.
    However, in the inference case, previously trained transformers are loaded.
    This function is a simplified pipeline; in a real context some of the steps
    would be in their own function or even module.
    Additionally, note that all this could be packed into a sklearn Pipeline.

    Input:
            df (data frame): dataset
            response (string): string of response name
                optional argument that could be used
                for naming variables or index y column
            artifact_path (string): path where artifacts are located
            train (bool): whether transformer fitting needs to be performed
    Output:
            X (data frame): X features
            y (data frame / series): y target
    '''
    processing_params = dict()
    if not train:
        # Load dataset processing parameters from training
        # - cols_cat
        # - cols_num
        # - num_features
        # - mean_imputer
        # - mode_imputer
        # - category_encoder
        try:
            # Check models can be loaded
            processing_params = joblib.load(artifact_path+'/processing_params.pkl')
        except FileNotFoundError:
            #print("The processing parameters from previous training not found!")
            logging.error("perform_data_processing: The processing parameters from previous training not found.")

    # New Churn variable (target): 1 Yes, 0 No
    col_org_response = 'Attrition_Flag'
    response_value_org_negative = "Existing Customer"
    try:
        assert col_org_response in data.columns
        data[response] = data[col_org_response].apply(
            lambda val: 0 if val == response_value_org_negative else 1)
    except AssertionError as err:
        #print(f"The df must contain the column '{col_org_response}'.")
        logging.error("perform_data_processing: The df must contain the column %s.", col_org_response)
        raise err
    except KeyError as err:
        #print("Response key must be a string!")
        logging.error("perform_data_processing: Response key must be a string.")
        raise err

    # Drop unnecessary columns
    cols_drop = [col_org_response, 'Unnamed: 0', 'CLIENTNUM'] # , response]
    try:
        for col in cols_drop:
            data.drop(col, axis=1, inplace=True)
    except KeyError as err:
        #print("Missing columns in the dataframe.")
        logging.error("perform_data_processing: Missing columns in the dataframe.")
        raise err

    # Drop duplicates
    data = data.drop_duplicates()

    # Automatically detect categorical columns
    cols_cat = list(data.select_dtypes(['object']).columns)
    # Automatically detect numerical columns
    cols_num = list(data.select_dtypes(['int64','float64']).columns)

    if train:
        # Persist categorical and numerical column names
        processing_params['cols_cat'] = cols_cat
        processing_params['cols_num'] = cols_num
    else:
        # Basic data check (deterministic):
        # Test that the automatically detected columns
        # match the ones detected during training
        col_lists = (cols_cat, cols_num)
        col_list_names = ('cols_cat', 'cols_num')
        for i in range(len(col_lists)):
            for col in col_lists[i]:
                try:
                    assert col in processing_params[col_list_names[i]]
                except AssertionError as err:
                    #print(f"Column {col} not found in set of columns from training.")
                    logging.error("perform_data_processing: Column not present in training dataset: %s.", col)
                    raise err

    # Handle missing values
    # - target: remove entries is target is NA
    # - numerical: mean
    # - categorical: mode
    data.dropna(subset=[response], axis=0, inplace=True)
    if train:
        # Create imputers, fit, transform data and persist
        mean_imputer = MeanImputer(variables=cols_num)
        mode_imputer = ModeImputer(variables=cols_cat)
        data = mean_imputer.fit_transform(data)
        data = mode_imputer.fit_transform(data)
        processing_params['mean_imputer'] = mean_imputer
        processing_params['mode_imputer'] = mode_imputer
    else:
        # Get trained imputers and transform data
        mean_imputer = processing_params['mean_imputer']
        mode_imputer = processing_params['mode_imputer']
        data = mean_imputer.transform(data)
        data = mode_imputer.transform(data)

    # Encode categorical variables as category ratios
    cols_cat_encoded = []
    if train:
        # Create encoder, fit, transform data and persist
        category_encoder = CategoryEncoder(features=cols_cat, target=response)
        data = category_encoder.fit_transform(data)
        processing_params['category_encoder'] = category_encoder
    else:
        # Get trained encoder and transform data
        category_encoder = processing_params['category_encoder']
        data = category_encoder.transform(data)
        cols_cat_encoded = category_encoder.encoded_categoricals_

    # Store target in y
    # and drop target from X <- data.
    # We cannot drop it beforehand because the encoding of the categoricals
    # needs to use y
    y = data[response]
    data.drop(response, axis=1, inplace=True)

    # Automatically detect numerical columns, AGAIN
    cols_num_encoded = list(data.select_dtypes(['int64','float64']).columns)

    # Features: categorical + numerical
    # BUT all identified as numerical now,
    # because we encoded them so!
    for col in cols_cat_encoded:
        try:
            assert col in cols_num_encoded
        except AssertionError as err:
            #print(f"Column {col} not found in set of numerical columns.")
            logging.error("perform_data_processing: Encoded column is not numerical: %s.", col)
            raise err
    cols_keep = cols_num_encoded
    # Build X
    X = data[cols_keep]

    # Basic data check (deterministic): Number of final features
    if train:
        processing_params['num_features'] = X.shape[1] # 19
    else:
        num_features = processing_params['num_features']
        try:
            assert X.shape[1] == num_features
        except AssertionError as err:
            #print(f"Wrong number of columns: {X.shape[1]} != {str(num_features)}")
            logging.error("perform_data_processing: Number of columns is different than in training dataset: %d != %d.", X.shape[1], num_features)
            raise err

    if train:
        # Persist dataset processing parameters from training
        # - cols_cat
        # - cols_num
        # - num_features
        # - mean_imputer
        # - mode_imputer
        # - category_encoder
        joblib.dump(processing_params, artifact_path+'/processing_params.pkl')

    logging.info("perform_data_processing: SUCCESS!")

    return X, y

def classification_report_image(y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds,
                                output_path):
    '''Produces classification report for training and testing results and stores report as image
    in images folder.

    Input:
            y_train (pandas.DataFrame): training response values
            y_test (pandas.DataFrame):  test response values
            y_train_preds_lr (np.array): training preds. from log. regression
            y_train_preds_rf (np.array): training preds. from random forest
            y_test_preds_lr (np.array): test preds. from logistic regression
            y_test_preds_lr (np.array): test preds. from logistic regression
            y_test_preds_rf (np.array): test preds. from random forest
            output_path (string): path to store the result figures
    Output:
            None
    '''
    # General parameters
    dpi = 200
    figsize = (15, 15)

    # Unpack
    y_train_preds_lr, y_train_preds_rf = y_train_preds
    y_test_preds_lr, y_test_preds_rf = y_test_preds

    # Random forest model: Classification report
    fig = plt.figure(figsize=figsize)
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    fig.savefig(output_path+'/rf_classification_report.png', dpi=dpi)

    # Logistic regression model: Classification report
    fig = plt.figure(figsize=figsize)
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    fig.savefig(output_path+'/lr_classification_report.png', dpi=dpi)

    logging.info("classification_report_image: SUCCESS!")

def roc_curve_plot(models, X_test, y_test, output_path):
    '''Creates and stores the feature importances in pth

    Input:
            models (objects): trained models
            X_test (data frame): test split features to compute ROC curves
            y_test (data frame / series): test split target to compute ROC curves
            output_path (string): path to store the result figure
    Output:
            None
    '''
    # General parameters
    dpi = 200
    figsize = (15, 8)

    # Unpack models
    lrc, rfc = models # lrc, cv_rfc.best_estimator_

    # ROC Plots
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    rfc_plot = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    fig.savefig(output_path+'/roc_plots.png', dpi=dpi)

    logging.info("roc_curve_plot: SUCCESS!")

def feature_importance_plot(model, X_data, output_path, filename_label):
    '''Creates and stores the feature importances in pth.
    Note that the function assumes there is no addition of features in the model pipeline,
    e.g., with PolynomialFeatures(); if so, the current function must be modified.

    Input:
            model (model object): model object containing feature_importances_
            X_data (data frame): pandas dataframe of X values
            output_path (string): path to store the result figure
            filename_label (string): label to add to the plot filename
    Output:
            None
    '''
    # Calculate feature importances
    importances = model['model'].feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    try:
        assert X_data.shape[1] == len(importances)
    except AssertionError as err:
        logging.error("feature_importance_plot: Number of dataset and model features differ: %d != %d.", X_data.shape[1], len(importances))
        logging.info("feature_importance_plot: Maybe PolynomialFeatures were used?")
        raise err

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    fig = plt.figure(figsize=(20,10))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save plot
    fig.savefig(output_path+'/feature_importance_'+filename_label+'.png', dpi=600)

    logging.info("feature_importance_plot: SUCCESS!")

def train_models(X_train,
                 y_train,
                 model_output_path="./models"):
    '''Trains and stores models.
    Instead of a model, a pipeline is stored,
    which contains some basic transformations (scaling, etc.).

    Several models are trained and stored:
    - Logistic regression
    - Random forests

    A grid search using cross-validation is carried out for each model.

    Note, however, that the transformations done in perform_data_processing()
    are obligatory: that function should precede the current.

    Input:
            X_train (data frame): X training data
            y_train (data frame / series): y training data
            model_output_path (string): path where models are stored
    Output:
            models (objects tuple): best trained models, using grid search and cross-validation
    '''
    # Model 1: Logistic Regression
    # Note: if the default solver='lbfgs' fails to converge, use another
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lr_pipe = Pipeline([
        ("polynomial_features", PolynomialFeatures()),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(solver='liblinear', random_state=42, max_iter=3000))])

    # Model 2: Random Forest Classifier
    # Note: scaling is really not necessary for random forests...
    # Polynomial features removed to plot feature importances
    rf_pipe = Pipeline([
        #("polynomial_features", PolynomialFeatures()),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42))])

    # Grid Search: Logistic Regression (Model 1)
    # Since we use polynomial features,
    # we cannot easily plot feature importances in the current implementation
    params_grid_lr = {
        'polynomial_features__degree': [1, 2],
        'model__penalty': ['l1', 'l2'],
        'model__C': np.geomspace(1e-3, 10, 6) # logarithmic jumps
    }
    lr_pipe_cv = GridSearchCV(estimator=lr_pipe, param_grid=params_grid_lr, cv=5)
    t1 = time.time()
    lr_pipe_cv.fit(X_train, y_train)
    # Get best logistic regression model/pipeline
    lr_pipe_cv_best = lr_pipe_cv.best_estimator_
    t2 = time.time()
    #print(f"Logistic regression trained with grid search and cross validation in {t2-t1:.2f} sec.")
    logging.info("train_models: Logistic regression trained in %.2f secs.", t2-t1)

    # Grid Search: Random Forest (Model 2)
    params_grid_rf = {
        #'polynomial_features__degree': [1], # [1, 2]
        'model__n_estimators': [200, 500],
        'model__max_features': ['auto', 'sqrt'],
        'model__max_depth': [4, 5, 100],
        'model__criterion': ['gini', 'entropy']
    }
    rf_pipe_cv = GridSearchCV(estimator=rf_pipe, param_grid=params_grid_rf, cv=5)
    t1 = time.time()
    rf_pipe_cv.fit(X_train, y_train)
    # Get best random forest model/pipeline
    rf_pipe_cv_best = rf_pipe_cv.best_estimator_
    t2 = time.time()
    #print(f"Random forest trained with grid search and cross validation in {t2-t1:.2f} sec.")
    logging.info("train_models: Random forest trained in %.2f secs.", t2-t1)

    # Save best models/pipelines
    joblib.dump(lr_pipe_cv_best, model_output_path+'/logistic_regression_model_pipe.pkl')
    joblib.dump(rf_pipe_cv_best, model_output_path+'/random_forest_model_pipe.pkl')

    models = (lr_pipe_cv_best, rf_pipe_cv_best)

    logging.info("train_models: SUCCESS!")

    return models

def evaluate_models(X_train,
                    X_test,
                    y_train,
                    y_test,
                    models,
                    eval_output_path="./images/results"):
    '''Evaluates the models.
    The models must have been already created by train_models()
    and are passed to the current function.

    Input:
            X_train (data frame): X training data
            X_test (data frame): X test data
            y_train (data frame / series): y training data
            y_test (data frame / series): y test data
            models (tuple of objects): trained models/pipelines
            eval_output_path (string): path where evaluation report images are stored
    Output:
            None
    '''
    # Unpack models
    lrc, rfc = models

    # Predict target for train & test features
    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Generate and save classification report plots
    classification_report_image(y_train,
                                y_test,
                                (y_train_preds_lr,y_train_preds_rf),
                                (y_test_preds_lr,y_test_preds_rf),
                                eval_output_path)

    # Save ROC curve plots
    roc_curve_plot(models, # (lrc, rfc)
                   X_test,
                   y_test,
                   eval_output_path)

    # Save plots of feature importance
    # IMPORTANT: If polynomial features are used,
    # feature importances don't work in the current implementation!
    filename_label = "random_forest"
    feature_importance_plot(rfc, X_train, eval_output_path, filename_label)
    #filename_label = "logistic_regression"
    #feature_importance_plot(lrc, X_train, eval_output_path, filename_label)

    logging.info("evaluate_models: SUCCESS!")

def load_model_pipeline(model_path):
    '''Loads model pipeline from path.'''
    try:
        # Load model with file and check this is successful
        model = joblib.load(model_path)
        logging.info("load_model_pipeline: SUCCESS!")
        return model
    except FileNotFoundError:
        #print("Model pipeline not found!")
        logging.error("load_model_pipeline: Model pipeline not found: %s.", model_path)
        return None

def predict(model_pipeline, X):
    '''Uses the model/pipeline to score the feature vectors.'''
    preds = model_pipeline.predict(X)

    return preds

def split(X, y):
    '''Splits X and y into train/test subsets.'''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

    return X_train, X_test, y_train, y_test

def run_training(config):
    '''Executes the complete model/pipeline generation:
    - Import dataset
    - Exploratory Data Analysis (EDA)
    - Data Cleaning and Feature Engineering (i.e., data processing)
    - Define and train models

    This is an example function that carries out all the model generation pipeline.
    In a real context we would use this template function to write our own,
    passing all the paths and configuration parameters from a config file.

    Input:
            None
    Output:
            None
    '''
    print("\n### TRAINING PIPELINE ###")

    # Load dataset
    # If not specified explicitly,
    # a sample with the first 10 entries (_sample.csv)
    # is saved for testing inference
    DATASET_PATH = config["dataset_path"] # "./data/bank_data.csv"
    print(f"\nLoading dataset...\t{DATASET_PATH}")
    df = import_data(DATASET_PATH, save_sample=True)

    # Perform Exploratory Data Analysis (EDA)
    # EDA report images are saved to `images/eda`
    print("\nPerforming EDA...")
    EDA_OUTPUT_PATH = config["eda_output_path"] # "./images/eda"
    perform_eda(df, EDA_OUTPUT_PATH)

    # Perform Feature Engineering
    # Artifacts like transformation objects, detected features & co.
    # are saved to `./artifacts`
    # IMPORTANT: train=True
    print("\nPerforming Data Processing...")
    RESPONSE = config["response"] # "Churn" # Target name
    ARTIFACT_PATH = config["artifact_path"] # "./artifacts"
    X, y = perform_data_processing(df, response=RESPONSE, artifact_path=ARTIFACT_PATH, train=True)

    # Train/Test split
    print("\nPerforming Train/Test Split...")
    X_train, X_test, y_train, y_test = split(X, y)

    # Train the models
    # Models are saved to `models/`
    print("\nTrainig...")
    MODEL_OUTPUT_PATH = config["model_output_path"] # "./models"
    models = train_models(X_train,
                          y_train,
                          model_output_path=MODEL_OUTPUT_PATH)

    # Evaluate the models
    # Report images saved to `images/results`
    print("\nEvaluating...")
    EVAL_OUTPUT_PATH = config["eval_output_path"] # "./images/results"
    evaluate_models(X_train,
                    X_test,
                    y_train,
                    y_test,
                    models,
                    eval_output_path=EVAL_OUTPUT_PATH)

    logging.info("run_training: SUCCESS!")

def run_inference(config):
    '''Executes an exemplary inference.
    The artifacts generated in the function run_training() are used here.

    This is an example function that runs the inference steps / pipeline.
    In a real context we would use this template function to write our own,
    passing all the paths and configuration parameters from a config file.
    Additionally, model serving requires having the model/pipeline in memory
    and answering to requests, which is not done here.

    Input:
            None
    Output:
            None
    '''
    print("\n### INFERENCE PIPELINE ###")

    # Load sample dataset
    DATASET_PATH = config["dataset_path_sample"] # "./data/bank_data_sample.csv"
    print(f"\nLoading exemplary dataset...\t{DATASET_PATH}")
    df = import_data(DATASET_PATH, save_sample=False)

    # Load model pipeline
    MODEL_FILENAME = config["inference_model_filename"] # "./models/random_forest_model_pipe.pkl"
    print(f"\nLoading model pipeline...\t{MODEL_FILENAME}")
    model_pipeline = load_model_pipeline(MODEL_FILENAME)

    # Perform Feature Engineering
    # Artifacts like transformation objects, detected features & co.
    # are read from `./artifacts`
    # IMPORTANT: train=False
    print("\nPerforming Data Processing...")
    RESPONSE = config["response"] # "Churn" # Target name
    ARTIFACT_PATH = config["artifact_path"] # "./artifacts"
    X, _ = perform_data_processing(df, response=RESPONSE, artifact_path=ARTIFACT_PATH, train=False)

    # Predict
    print("\nInference...")
    y_pred = predict(model_pipeline, X)
    print("Sample index and prediction value pairs:")
    print(list(zip(list(df.index),list(y_pred))))

    logging.info("run_inference: SUCCESS!")

if __name__ == "__main__":
    '''Two pipelines are executed one after the other:
    (1) model generation/training
    (2) and exemplary inference.

    If the models have been generated (pipeline 1), we can comment its call out
    and simply run the inference (pipeline 2).
    '''

    # Load the configuration file
    config_filename = "config.yaml"
    #config_filename = "customer_churn/config.yaml"
    with open(config_filename, 'r') as stream:
        config = yaml.safe_load(stream)

    # Pipeline 1: Data Analysis and Modeling
    # Dataset is loaded and analyzed; models and artifacts are created.
    run_training(config)

    # Pipeline 2: Exemplary Inference
    # Sample dataset, trained models and artifacts are loaded;
    # an exemplary inference is carried out.
    run_inference(config)
