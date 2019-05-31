#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:37:30 2019

@authors: Ryan Clark, Matt Hong, So Sosaki

File Description:
This is the main program that runs all of the models against the data. Each
model that is ran through here is assumed to have a separate classifier and
regressor even if at times this is uncessary for the particular model. Each
classifier and regressor is assumed to be able to handle multiple labels.
Please view scripts/models.py if you need assistance in understanding this 
better. This file report four types of errors (see errors.py for more 
detail) in both functions. If run_model_phone_id is ran, 20 plots will be saved
as well as the errors for each unique phone id. This is useful if the motion of
each phone is wanting to be tracked over time (see plots.py for more detail).
The run_model function simply runs the model on the data in its entirety. It
will return and save the total report but no figures. It is ~6 seconds faster
than the phone_id function. It is recommended to just run the run_model
function if you just want total outputs.

Timers off of 2016 Macbook Pro: 
         K-Nearest Neighbors    ~30 seconds
         Random Forrest         ~160 seconds
         Support Vector Machine ~12 seconds

File Structure:
    main.py
    data/
        trainingData.csv
        validationData.csv
    output/
        *
    scripts/
        *
"""

# Scripts
from scripts.utils import (load_data, save_fig, create_subreport, save_report,
                           filter_out_low_WAPS)
from scripts.errors import compute_errors
from scripts.plots import plot_pos_vs_time, plot_lat_vs_lon
from scripts.models import (load_KNN, load_Random_Forest, load_SVM,
                           load_Decision_Tree, threshold_variance, pca)

# Libraries
from time import time
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import close, ioff, ion
from pandas import DataFrame, concat

# Hyper-parameters / CONSTANTS
N = 520 # Number of WAPS - CONSTANT
MIN_WAPS = 9 # Required number of active WAPS per sample.
NO_SIGNAL_VALUE = -98 # Changed Null Value
QUANTITATIVE_COLUMNS = ['LONGITUDE', 'LATITUDE'] # Regression Columns
CATEGORICAL_COLUMNS = ['FLOOR', 'BUILDINGID'] # Classification Columns
DROP_VAL = True # if True, drops the validation dataset which may be corrupted
# Used to remove columns where information is missing the validation data.
DROP_COLUMNS =["SPACEID" ,"RELATIVEPOSITION", "USERID"]
SAVE_FIGS = True # Trigger to save/overwrite figures(saves 5 seconds if False)
SAVE_REPORT = True # Trigger to save/overwrite report
PRINT_SUB = True # Trigger to print sub reports or not.
DISPLAY_PLOTS = False # If true, the 20 figures will be created on screen.


def run_model_phone_id(model_name, clf, regr, data):
    '''
    This function runs the model after grouping the data into separate phone
    id groups. It will also report the total error as well as the error 
    for each phone id. This will also generate 2 sets of 10 plots where one is
    lat vs. long and the other is pos vs. timestamp. This may or may not save
    the data depending on the specified parameters.
    
    Parameters: model_name : (str)
                clf        : classifier with fit and predict class functions
                regr       : regressor with fit and predict class functions
                data       : (tuple) of the 4 sets of data
                
    Returns:    report     : (str) contains all error information
    '''
    tic_model = time() # Start model performance timer

    x_train, x_test, y_train, y_test = data # Decompose tuple into datasets

    # Initialize total error tracks to report at end of function
    tcoords_error = list()
    tbuild_missclass = tfloor_missclass = tstandard_error = 0
    subreports = list()    

    clf_fit = clf.fit(x_train, y_train[CATEGORICAL_COLUMNS])                
    regr_fit = regr.fit(x_train, y_train[QUANTITATIVE_COLUMNS])
    
    # Loop through each phone_id, group them, compute and report errors.
    for phone_id, y_test_group in y_test.groupby("PHONEID", sort=False):
        
        # Obtain respective data for the phone_id labels (y_test_group)
        x_test_group = x_test[y_test_group.index]
        
        prediction = clf_fit.predict(x_test_group)
        clf_pred = DataFrame(prediction, columns=CATEGORICAL_COLUMNS)

        prediction = regr_fit.predict(x_test_group)
        regr_pred = DataFrame(prediction, columns=QUANTITATIVE_COLUMNS)
        
        prediction = concat((clf_pred, regr_pred), axis=1)
        
        errors = compute_errors(prediction, y_test_group)
                
        # Format the errors into a string with headers for each error type.
        # Printing can be disabled due to noise. Append for final report.
        subreport = create_subreport(errors, y_test_group.shape[0], phone_id)
        if PRINT_SUB:
            print(subreport + '\n')
        subreports.append(subreport)
        
        # Take the phone_id errors and combine them with the running totals.
        build_missclass, floor_missclass, coords_error, std_error, _ = errors
        tcoords_error.extend(coords_error)
        tbuild_missclass += build_missclass
        tfloor_missclass += floor_missclass
        tstandard_error += std_error
        
        # Plot pos vs. time and lat vs long. Save plots if boolean permits
        fig = plot_pos_vs_time(prediction, y_test_group, phone_id, model_name)
        if SAVE_FIGS:
            save_fig(fig, model_name, phone_id, "Position_vs_Timestamp")
        if not DISPLAY_PLOTS:
            close(fig)
        
        fig = plot_lat_vs_lon(prediction, y_test_group, phone_id, model_name)
        if SAVE_FIGS:
            save_fig(fig, model_name, phone_id, "Latitude_vs_Longitude")    
        if not DISPLAY_PLOTS:
            close(fig)
            
    # Compute totals report and print it
    terrors = (tbuild_missclass, tfloor_missclass, tcoords_error, 
               tstandard_error, "N/A")
    totals_report = create_subreport(terrors, y_test.shape[0])
    print(totals_report + '\n')
    
    toc_model = time()
    model_timer = toc_model - tic_model
    print("%s Timer: %.2f seconds" % (model_name, model_timer))
    
    # Create the output txt file of the entire report. Save if boolean permits.
    header = "%s\nModel Timer: %.2f seconds" % (model_name, model_timer)
    sub_reports = "\n\n".join(subreports)
    report = "\n\n".join([header, totals_report, sub_reports])
    if SAVE_REPORT:
        save_report(model_name, report, "phone_ids")
    
    return report
    

def run_model(model_name, clf, regr, data):
    '''
    This runs the input model (classifier and regressor) against the dataset
    and prints out the error report.
    
    Parameters: model_name : (str)
                clf        : classifier with fit and predict class functions
                regr       : regressor with fit and predict class functions
                data       : (tuple) of the 4 sets of data
                
    Returns:    errors     : (tuple) contains all error information
                prediction : (DataFrame) prediction of y_test
    '''
    tic_model = time() # Start model performance timer

    x_train, x_test, y_train, y_test = data # Decompose tuple into datasets

    # Classifier
    fit = clf.fit(x_train, y_train[CATEGORICAL_COLUMNS])
    prediction = fit.predict(x_test)
    clf_prediction = DataFrame(prediction, columns=CATEGORICAL_COLUMNS)
              
    # Regressor
    fit = regr.fit(x_train, y_train[QUANTITATIVE_COLUMNS])
    prediction = fit.predict(x_test)
    regr_prediction = DataFrame(prediction, columns=QUANTITATIVE_COLUMNS)
    
    prediction = concat((clf_prediction, regr_prediction), axis=1)
    
    errors = compute_errors(prediction, y_test)
    
    # Compute totals report and print it
    totals_report = create_subreport(errors, y_test.shape[0])
    print(totals_report)
    
    toc_model = time()
    model_timer = toc_model - tic_model
    print("%s Timer: %.2f seconds\n" % (model_name, model_timer))
    
    # Create the output txt file of the entire report. Save if boolean permits.
    header = "%s\nModel Timer: %.2f seconds" % (model_name, model_timer)
    report = "\n\n".join([header, totals_report])
    if SAVE_REPORT:
        save_report(model_name, report, "totals")
    
    return errors, prediction
    
################################## MAIN #######################################
    
if __name__ == "__main__":

    tic = time() # Start program performance timer
    
    close("all") # Close all previously opened plots
    
    ion() if DISPLAY_PLOTS else ioff()
    
    # Load and preprocess data with all methods that are independent of subset.
    data = load_data("trainingData.csv", "validationData.csv", N, DROP_COLUMNS,
                     dst_null=NO_SIGNAL_VALUE, drop_val=DROP_VAL)    
    X, Y = data                
    
    # Note that Random Seed is 0. All Validation sets must be created from a
    # subset of the train set here.
    x_train_o, x_test_o, y_train, y_test = train_test_split(X.values, Y.values, 
                                             test_size=0.2, random_state=0)
    
    # This filters out samples that do not have enough actie WAPs in it 
    # according to MIN_WAPS. This has to happen after the split because if not,
    # the randomness will be affected by missing samples, thus compromising 
    # test set validity.
    x_train_o, y_train = filter_out_low_WAPS(x_train_o, y_train, MIN_WAPS)
    x_test_o, y_test = filter_out_low_WAPS(x_test_o, y_test, MIN_WAPS)

    y_train = DataFrame(y_train, columns=Y.columns)
    y_test = DataFrame(y_test, columns=Y.columns)
    
    ################## INSERT MODEL AND MODEL NAME HERE #######################
    
    # K-Nearest Neighbors with Variance Thresholding
    model_name, clf, regr = load_KNN()
    x_train, x_test = threshold_variance(x_train_o, x_test_o, thresh=0.00001)
    data_in =  (x_train, x_test, y_train, y_test)
    knn_errors, knn_prediction = run_model(model_name, clf, regr, data_in)
    knn_report = run_model_phone_id(model_name, clf, regr, data_in)
    
    # Decision Tree with PCA
    model_name, clf, regr= load_Decision_Tree()
    x_train, x_test = pca(x_train_o, x_test_o, perc_of_var=0.95)
    data_in =  (x_train, x_test, y_train, y_test)
    dt_errors, dt_prediction = run_model(model_name, clf, regr, data_in)
    dt_report = run_model_phone_id(model_name, clf, regr, data_in)

    # Random Forest with PCA
    model_name, clf, regr= load_Random_Forest()
    x_train, x_test = pca(x_train_o, x_test_o, perc_of_var=0.95)
    data_in =  (x_train, x_test, y_train, y_test)
    rf_errors, rf_prediction = run_model(model_name, clf, regr, data_in)
    rf_report = run_model_phone_id(model_name, clf, regr, data_in)
    
    # Support Vector Machine with PCA
    model_name, clf, regr = load_SVM()
    x_train, x_test = pca(x_train_o, x_test_o, perc_of_var=0.95)
    data_in =  (x_train, x_test, y_train, y_test)
    svm_errors, svm_prediction = run_model(model_name, clf, regr, data_in)
    svm_report = run_model_phone_id(model_name, clf, regr, data_in)

    ################# INSERT MODEL AND MODEL NAME HERE ########################
    
    toc = time() # Report program performance timer
    print("Program Timer: %.2f seconds" % (toc-tic))