'''This module performs the necessary data science processing steps
to build and save models that predict customer churn using the
Credit Card Customers dataset from Kaggle:

https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code

These are the steps which are carried out:
- The dataset is loaded
- Exploratory Data Analysis (EDA)
- Feature Engineering (FE)
- Training: Random Forest and Logistic Regression models are fit
- Classification report plots

Clean code principles are guaranteed:
- Modularized code
- PEP8 conventions
- Error handling
- Testing is carried in the companion file: test_churn_library.py

PEP8 conventions checked with:

>> pylint churn_library.py # 8.30/10
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
#os.environ['QT_QPA_PLATFORM']='offscreen'
import joblib

#import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # It needs to be called here, otherwise we get error!
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set()

#from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

from transformations import MeanImputer, ModeImputer, CategoryEncoder

# Set library/module options
os.environ['QT_QPA_PLATFORM']='offscreen'
matplotlib.use('TkAgg')
sns.set()

def import_data(pth, save_sample=True):
    '''Returns dataframe for the CSV found at pth.
    For training save_sample=True; for inference save_sample=False.
    The previously saved data sample contains the first 10 entries
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
        return data
    except (FileNotFoundError, NameError):
        print("File not found!")
    except IsADirectoryError:
        print("File is a directory!")

    return None

def perform_eda(data, output_path):
    '''Performs EDA on df and saves figures to the output_path folder.
    
    Input:
            df (pandas.DataFrame): dataset
            output_path (string): path where report images are saved

    Output:
            None
    '''
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


def encoder_helper(data, category_lst, response="Churn"):
    '''Helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15.
    
    Input:
            df (pandas.DataFrame): dataset
            category_lst (list of str): list of columns that contain
                categorical features
            response (str): string of response name [optional argument
                that could be used for naming variables or index y column]

    Output:
            df (pandas.DataFrame): pandas dataframe with new columns
            cat_columns_encoded (list of str): names of new columns
    '''

    # Names of new encoded columns
    cat_columns_encoded = []

    # Automatically detect categorical columns
    if not category_lst:
        category_lst = list(data.select_dtypes(['object']).columns)

    # Loop over all categorical columns
    # Create new variable which contains the churn ratio
    # associated with each category
    for col in category_lst:
        col_lst = []
        col_groups = data.groupby(col).mean()[response]

        for val in data[col]:
            col_lst.append(col_groups.loc[val])

        col_encoded_name = col + "_" + response
        cat_columns_encoded.append(col_encoded_name)
        data[col_encoded_name] = col_lst

    return data, cat_columns_encoded


def perform_feature_engineering(data, response="Churn", artifact_path="./artifacts", train=True):
    '''
    Perform Feature Engineering: 
    - basic cleaning,
    - select/drop features,
    - encode categoricals,
    - data checks, 
    - split.
    
    This is a simplified pipeline; in a real context some of the steps
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
    
    
    # New Churn variable (target): 1 Yes, 0 No
    try:
        assert 'Attrition_Flag' in data.columns
        data[response] = data['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
    except AssertionError as err:
        print("The df must contain the column 'Attrition_Flag'.")
        raise err
    except KeyError as err:
        print("Response key must be a string!")
        raise err
    
    # Drop unnecessary columns
    cols_drop = ['Attrition_Flag', 'Unnamed: 0', 'CLIENTNUM'] # , response]
    try:
        for col in cols_drop:
            data.drop(col, axis=1, inplace=True)    
    except KeyError as err:
        print("Missing columns in the dataframe.")
        raise err

    # Drop duplicates

    # Automatically detect categorical columns
    cols_cat = list(data.select_dtypes(['object']).columns)

    # Automatically detect numerical columns
    cols_num = list(data.select_dtypes(['int64','float64']).columns)
    
    # Handle missing values
    # - target: remove entry
    # - numerical: mean
    # - categorical: mode

    # Encode categorical variables as category ratios
    cat_columns_encoded = []
    data, cat_columns_encoded = encoder_helper(data, cols_cat, response)

    # Store target in y
    # and drop target from X <- data.
    # We cannot drop it beforehand because the encoding of the categoricals
    # needs to use y
    y = data[response]
    data.drop(response, axis=1, inplace=True)

    # Automatically detect numerical columns, AGAIN
    cols_num = list(data.select_dtypes(['int64','float64']).columns)

    # Features: categorical + numerical
    # BUT all identified as numerical now,
    # because we encoded them so!
    for col in cat_columns_encoded:
        try:
            assert col in cols_num
        except AssertionError as err:
            print(f"Column {col} not found in set of numerical columns.")
            raise err
    cols_keep = cols_num
    # Build X
    X = data[cols_keep]

    # Basic data checks (deterministic)
    try:
        assert X.shape[1] == 19
    except AssertionError as err:
        print(f"Wrong number of columns: {X.shape[1]} != 19")
        raise err

    return X, y

def classification_report_image(y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds,
                                output_path):
    '''
    Produces classification report for training and testing results and stores report as image
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

def roc_curve_plot(models, X_test, y_test, output_path):
    '''
    Creates and stores the feature importances in pth
    
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

def feature_importance_plot(model, X_data, output_path):
    '''
    Creates and stores the feature importances in pth.
    
    Input:
            model (model object): model object containing feature_importances_
            X_data (data frame): pandas dataframe of X values
            output_path (string): path to store the result figure

    Output:
            None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    fig = plt.figure(figsize=(20,10))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labelss
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save plot
    fig.savefig(output_path+'/feature_importance.png', dpi=600)

def train_models(X_train, X_test, y_train, y_test, eval_output_path="./images/results", model_output_path="./models"):
    '''
    Train, store model results: images + scores, and store models.
    
    Input:
            X_train (data frame): X training data
            X_test (data frame): X testing data
            y_train (data frame / series): y training data
            y_test (data frame / series): y testing data
            eval_output_path (string): path where evaluation report images are stored
            model_output_path (string): path where models are stored
            
    Output:
            None
    '''
    # Grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Save best model
    joblib.dump(cv_rfc.best_estimator_, model_output_path+'/rfc_model_best.pkl')
    joblib.dump(cv_rfc, model_output_path+'/rfc_model.pkl')
    joblib.dump(lrc, model_output_path+'/logistic_model.pkl')

    #try:
    #    # Check models can be loaded
    #    rfc_model = joblib.load('./models/rfc_model_best.pkl')
    #    lr_model = joblib.load('./models/logistic_model.pkl')
    #except FileNotFoundError:
    #    print("Models could not be saved!")

    # Save classification report plots
    classification_report_image(y_train,
                                y_test,
                                (y_train_preds_lr,y_train_preds_rf),
                                (y_test_preds_lr,y_test_preds_rf),
                                eval_output_path)

    # Pack models for the ROC curve computation
    models = (lrc, cv_rfc.best_estimator_)

    # Save ROC curve plots
    roc_curve_plot(models,
                   X_test,
                   y_test,
                   eval_output_path)

    # Save plot of feature importance
    feature_importance_plot(cv_rfc, X_train, eval_output_path)

def run_analysis_and_training():
    '''
    Execute the complete model/pipeline generation:
    - Import dataset
    - Exploratory Data Analysis (EDA)
    - Data Cleaning and Feature Engineering
    - Define and train models
    In a real context we would pass the path to the dataset.
    
    Input:
            None    
    Output:
            None
    '''    
    # Load dataset
    # If not specified explicitly, 
    # a sample with the first 10 entries (_sample.csv)
    # is saved for testing inference
    print("Loading dataset...")
    DATASET_PATH = "./data/bank_data.csv"
    df = import_data(DATASET_PATH, save_sample=True)

    # Perform Exploratory Data Analysis (EDA)
    # EDA report images are saved to `images/eda`
    print("Performing EDA...")
    EDA_OUTPUT_PATH = "./images/eda"
    perform_eda(df, EDA_OUTPUT_PATH)

    # Perform Feature Engineering
    # Artifacts like transformation objects, detected features & co.
    # are saved to `./artifacts`
    print("Performing Feature Engineering...")
    RESPONSE = "Churn" # Target name
    ARTIFACT_PATH = "./artifacts"
    X, y = perform_feature_engineering(df, response=RESPONSE, artifact_path=ARTIFACT_PATH, train=True)

    # Train/Test split
    print("Performing Train/Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

    # Train the models
    # Models are saved to `models/`
    # and report images saved to `images/results`
    print("Trainig...")
    EVAL_OUTPUT_PATH = "./images/results"
    MODEL_OUTPUT_PATH = "./models"
    train_models(X_train, X_test,
                 y_train, y_test,
                 eval_output_path=EVAL_OUTPUT_PATH,
                 model_output_path=MODEL_OUTPUT_PATH)

def run_inference():
    '''
    Execute the an exemplary inference.
    In a real context we would pass the path of the data to be inferred.
    Additionally, model serving requires having the model/pipeline in memory
    and answering to requests, which is not done here.
    The artifacts generated in the function run_analysis_and_training() are used here.
    
    Input:
            None    
    Output:
            None
    '''    
    # Load sample dataset
    print("Loading exemplary dataset...")
    DATASET_PATH = "./data/bank_data_sample.csv"
    df = import_data(DATASET_PATH, save_sample=False)

    # Load model pipeline

    # Perform Feature Engineering
    # Artifacts like transformation objects, detected features & co.
    # are read from `./artifacts`
    print("Performing Feature Engineering...")
    RESPONSE = "Churn" # Target name
    ARTIFACT_PATH = "./artifacts"
    X, _ = perform_feature_engineering(df, response=RESPONSE, artifact_path=ARTIFACT_PATH, train=False)

    # Predict

if __name__ == "__main__":

    # Pipeline 1: Data Analysis and Modeling
    # Dataset is loaded and analyzed; models and artifacts are created.
    run_analysis_and_training()
    
    # Pipeline 2: Exemplary Inference
    # Sample dataset, trained models and artifacts are loaded; an exemplary inference is carried out.
    run_inference()