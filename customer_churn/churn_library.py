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
    - Exemplary dataset is loaded (generated in Pipeline 1)
    - Model pipeline is loaded (generated in Pipeline 1)
    - Data Processing: Cleaning and Feature Engineering (FE) (same as in Pipeline 1)
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

>> pylint churn_library.py # 8.07/10
>> autopep8 churn_library.py

This file cannot be run stand-alone;
instead, we can use in a `main.py` script defined as follows:

```
from customer_churn import churn_library as churn

if __name__ == "__main__":
    config_filename="config.yaml"
    churn.run(config_filename)
```

Then, we execute the main file:

>> python main.py

The script expects the proper dataset to be located in `./data`
or the folder specified in `config.yaml`.

Additionally, `config.yaml` defines the storage locations of other elements,
with the following defaults:

- Any produced models: `./models`
- Any plots of the EDA and the classification: `./images`
- All other artifacts (e.g., data processing parameters): `./artifacts`

If those folders are not present, they are created automatically.

The logging output will be saved in a local *.log file.

Author: Mikel Sagardia
Date: 2022-06-08
'''

from codecs import escape_encode
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
from sklearn.preprocessing import PowerTransformer
from sklearn.utils import resample

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

#from transformations import MeanImputer, ModeImputer, CategoryEncoder
from .transformations import MeanImputer, ModeImputer, CategoryEncoder

# Set library/module options
os.environ['QT_QPA_PLATFORM']='offscreen'
matplotlib.use('TkAgg')
sns.set()

# Logging configuration
logging.basicConfig(
    filename='./churn_library.log', # filename, where it's dumped
    level=logging.INFO, # minimum level I log
    filemode='w',
    # https://docs.python.org/3/library/logging.html
    # logger - time - level - our message
    format='%(name)s - %(asctime)s - %(levelname)s - %(message)s')

def get_data(pth, save_sample=True):
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
            # Choose n_samples samples correctly stratified
            # according to the target class(es)
            data_sample = resample(df,
                                   n_samples=100,
                                   replace=False,
                                   stratify=df['Attrition_Flag'],
                                   random_state=0)
            #data_sample = data.iloc[:10,:]
            data_sample.to_csv(pth_sample, sep=',', header=True, index=False)
        logging.info("get_data: SUCCESS!")
        return data
    except (FileNotFoundError, NameError):
        #print("File not found!")
        logging.error("get_data: File not found: %s.", pth)
    except IsADirectoryError:
        #print("File is a directory!")
        logging.error("get_data: File is a directory.")

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
                            processing_params,
                            response="Churn",
                            artifact_path="./artifacts",
                            train=True):
    '''Performs basic Data Processing and Feature Engineering:
    - basic cleaning,
    - mappings,
    - select/drop features,
    - transform numericals,
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
            processing_params (object): dictionary with processing parameters
            response (string): string of response name
                optional argument that could be used
                for naming variables or index y column
            artifact_path (string): path where artifacts are located (processing params)
            train (bool): whether transformer fitting needs to be performed
    Output:
            X (data frame): X features
            y (data frame / series): y target
    '''
    # processing_params: dict()
    # - mappings
    # - cols_drop
    # - cols_cat
    # - cols_num
    # - num_features
    # - mean_imputer
    # - mode_imputer
    # - numerical_transformations
    # - category_encoder

    # Mappings
    # 1) Define/Get
    mappings = dict()
    if train:
        # Mapping 1
        map_col_1 = 'Attrition_Flag'
        mapping_1 = {'Existing Customer':0, 'Attrited Customer':1} # New Churn variable (target): 1 Yes, 0 No
        mappings[map_col_1] = mapping_1
        # Mapping 2
        # ...
        # Save to processing parameters
        processing_params['mappings'] = mappings
    else:
        mappings = processing_params['mappings']
    # 2) Apply
    for col, mapping in mappings:
        try:
            assert col in data.columns
            data[col].replace(mapping, inplace=True)
        except AssertionError as err:
            #print(f"The df must contain the column '{col}'.")
            logging.error("perform_data_processing: The df must contain the column %s.", col)
            raise err
        
    # New Churn variable (target): 1 Yes, 0 No
    col_org_response = 'Attrition_Flag'
    try:
        data[response] = data[col_org_response]
    except KeyError as err:
        #print("Response key must be a string!")
        logging.error("perform_data_processing: Response key (target) must be a string.")
        raise err

    # Drop unnecessary columns
    # 1) Define/Get
    cols_drop = []
    if train:
        # Maybe we can read this from the config.yaml
        cols_drop = ['Attrition_Flag', 'Unnamed: 0', 'CLIENTNUM'] # , response]
        processing_params['cols_drop'] = cols_drop
    else:
        cols_drop = processing_params['cols_drop']
    # 2) Apply
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

    # Persist categorical and numerical column names or check
    if train:
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
    # - numerical: impute mean
    # - categorical: impute mode
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

    # Transform numerical values if they are very skewed
    # Rule of thumb: abs(skew) > 0.75 -> apply (power) transformation
    numerical_transformations = dict()
    if train:
        for col in cols_num:
            if abs(data[col].skew()) > 0.75:
                pt = PowerTransformer('yeo-johnson', standardize=False)
                data[col] = pt.fit_transform(data[col].values.reshape(-1,1))
                numerical_transformations[col] = pt
        processing_params['numerical_transformations'] = numerical_transformations
    else:
        numerical_transformations = processing_params['numerical_transformations']
        for col, pt in numerical_transformations.items():
            try:
                assert col in data.columns
                data[col] = pt.transform(data[col].values.reshape(-1,1))
            except AssertionError as err:
                #print(f"The df must contain the column '{col}'.")
                logging.error("perform_data_processing: The df must contain the column %s.", col)
                raise err

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
        # - mappings
        # - cols_drop
        # - cols_cat
        # - cols_num
        # - num_features
        # - mean_imputer
        # - mode_imputer
        # - numerical_transformations
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

    # Model 2: Random Forest Classifier
    # Note: scaling is really not necessary for random forests...
    # Polynomial features removed to plot feature importances
    rf_pipe = Pipeline([
        #("polynomial_features", PolynomialFeatures()),
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42))])

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

def load_processing_params(params_path):
    '''Loads processing params from path.'''
    try:
        # Load processing params file and check this is successful
        processing_params = joblib.load(params_path)
        logging.info("load_processing_params: SUCCESS!")
        return processing_params
    except FileNotFoundError:
        #print("Processing params file not found!")
        logging.error("load_processing_params: Processing params file not found: %s.", params_path)
        return None

def predict(model_pipeline, X):
    '''Uses the model/pipeline to score the feature vectors.'''
    preds = model_pipeline.predict(X)

    return preds

def split(X, y):
    '''Splits X and y into train/test subsets.'''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test

def run_training(config, produce_images=True):
    '''Executes the complete model/pipeline generation:
    - Import dataset
    - Exploratory Data Analysis (EDA) (if produce_images=True)
    - Data Cleaning and Feature Engineering (i.e., data processing)
    - Define and train models
    - Evaluate models (if produce_images=True)

    This is an example function that carries out all the model generation pipeline.
    In a real context we would use this template function to write our own,
    passing all the paths and configuration parameters from a config file.

    Note that the EDA and the evaluation are optional and are
    triggered if produce_images=True (default).

    Input:
            config (dictionary): configuration parameters
            produce_images (boolean): whether to run EDA and evaluation and produce images
    Output:
            None
    '''
    print("\n### TRAINING PIPELINE ###")

    # Load dataset
    # If not specified explicitly,
    # a sample with the first 10 entries (_sample.csv)
    # is saved for testing inference
    DATASET_PATH = config["dataset_filename"] # "./data/bank_data.csv"
    print(f"\nLoading dataset...\t{DATASET_PATH}")
    df = get_data(DATASET_PATH, save_sample=True)

    if (produce_images):
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
    processing_params = dict() # initialize as empty dictionary to be filled
    X, y = perform_data_processing(df, processing_params, response=RESPONSE, artifact_path=ARTIFACT_PATH, train=True)

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

    if produce_images:
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
            config (dictionary): configuration parameters
    Output:
            None
    '''
    print("\n### INFERENCE PIPELINE ###")

    # Load/get sample dataset
    DATASET_PATH = config["dataset_sample_filename"] # "./data/bank_data_sample.csv"
    print(f"\nLoading exemplary dataset...\t{DATASET_PATH}")
    df = get_data(DATASET_PATH, save_sample=False)

    # Load processing parameters
    # This will be done only once in online inferences    
    PARAMS_FILENAME = config["processing_params_filename"] # "./artifacts/processing_params.pkl"
    print(f"\nLoading processing params...\t{PARAMS_FILENAME}")
    processing_params = load_processing_params(PARAMS_FILENAME)

    # Perform Feature Engineering
    # Artifacts like transformation objects, detected features & co.
    # are read from `./artifacts`
    # IMPORTANT: train=False
    print("\nPerforming Data Processing...")
    RESPONSE = config["response"] # "Churn" # Target name
    ARTIFACT_PATH = config["artifact_path"] # "./artifacts"
    X, _ = perform_data_processing(df, processing_params, response=RESPONSE, artifact_path=ARTIFACT_PATH, train=False)

    # Load model pipeline
    # This will be done only once in online inferences
    MODEL_FILENAME = config["inference_model_filename"] # "./models/random_forest_model_pipe.pkl"
    print(f"\nLoading model pipeline...\t{MODEL_FILENAME}")
    model_pipeline = load_model_pipeline(MODEL_FILENAME)

    # Predict
    print("\nInference...")
    y_pred = predict(model_pipeline, X)
    print("Sample index and prediction value pairs:")
    print(list(zip(list(df.index),list(y_pred))))

    logging.info("run_inference: SUCCESS!")

def run_setup(config_filename="config.yaml"):
    """Loads configuration file and check that all folders exist;
    if not, create them

    Input:
        config_filename (str, optional): path to the configuration file
            Defaults to "config.yaml"

    Output:
        config (dictionary): configuration parameters
    """
    # Load the configuration file
    try:
        with open(config_filename, 'r') as stream:
            config = yaml.safe_load(stream)
    except FileNotFoundError as err:
        logging.error("run_setup: Configuration file not found: %s.", config_filename)
        return None

    # Check folders
    # data_path: ./data
    # eda_output_path: ./images/eda
    # artifact_path: ./artifacts
    # model_output_path: ./models
    # eval_output_path: ./images/results
    folders = [config["data_path"],
               config["eda_output_path"],
               config["artifact_path"],
               config["model_output_path"],
               config["eval_output_path"]]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    return config

def run(config_filename="config.yaml", produce_images=True):
    '''This function runs the complete customer churn library.

    First, the setup is carried out:
    - the configuration file is loaded
    - and the existence of necessary folders is checked.

    Then, two pipelines are executed one after the other:
    (1) model generation/training
    (2) and exemplary inference.

    If the models have been generated (pipeline 1), we can comment its call out
    and simply run the inference (pipeline 2).
    '''

    # Load the configuration file and check/set folders
    config = run_setup(config_filename=config_filename)

    # Pipeline 1: Data Analysis and Modeling
    # Dataset is loaded and analyzed; models and artifacts are created.
    run_training(config, produce_images=produce_images)

    # Pipeline 2: Exemplary Inference
    # Sample dataset, trained models and artifacts are loaded;
    # an exemplary inference is carried out.
    run_inference(config)
