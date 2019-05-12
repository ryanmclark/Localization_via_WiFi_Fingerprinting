#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:37:30 2019

@author: Ryan Clark

File Description:
This file is built to handle any model that has the .fit and .predict class 
functions. If the models have these properties, then the program will
properly process the output and report it into the output/<model_name> 
directory. This file report four types of errors (see errors.py for more 
detail) in both functions. If run_model_phone_id is ran, 20 plots will be saved
as well as the errors for each unique phone id. This is useful if the motion of
each phone is wanting to be tracked over time (see plots.py for more detail).
The run_model function simply runs the model on the data in its entirety. It
will return and save the total report but no figures. It is ~6 seconds faster
than the phone_id function. This is useful when querying for parameters.

Note: The .predict class function must return a 2D array that is the same shape
      of the label data with the same content (y_train and y_test).

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
from scripts.utils import load_data, save_fig, create_subreport, save_report
from scripts.errors import compute_errors
from scripts.plots import plot_pos_vs_time, plot_lat_vs_lon

# Libraries
from time import time
 # EXAMPLE - IMPORT MODELS SIMILARLY - CUSTOM OR PREMADE
from scripts.neuralnet import NeuralNetworkRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from matplotlib.pyplot import close, ioff, ion
from pandas import DataFrame
from numpy import hstack

# Hyper-parameters
N = 520 # Number of WAPS - CONSTANT
QUANTITATIVE_COLUMNS = ['LONGITUDE', 'LATITUDE']
CATEGORICAL_COLUMNS = ['FLOOR', 'BUILDINGID']
# Used to remove columns where information is missing the validation data.
DROP_COLUMNS = ["SPACEID" ,"RELATIVEPOSITION", "USERID"]
SAVE_FIGS = True # Trigger to save/overwrite figures (saves 5 seconds if False)
SAVE_REPORT = True # Trigger to save/overwrite report
PRINT_SUB = True # Trigger to print sub reports or not.
DISPLAY_PLOTS = False # If true, the 20 figures will be created on screen.

def run_model_phone_id(model_name, model_regressor, model_classifier, data, fig_trig=SAVE_FIGS, 
                       rep_trig=SAVE_REPORT, print_trig=PRINT_SUB,
                       display=DISPLAY_PLOTS):
    '''
    This function runs the model after grouping the data into separate phone
    id groups. It will also report the total error as well as the error 
    for each phone id. This will also generate 2 sets of 10 plots where one is
    lat vs. long and the other is pos vs. timestamp. This may or may not save
    the data depending on the specified parameters.
    
    Parameters: model_name       : (str)
                model_regressor  : (*) must have .fit and .predict class functions.
                model_classifier : (*) must have .fit and .predict class functions.
                data             : (tuple) contains the 4 sets of data.
                fig_trig         : (boolean) if true saves and overwrites figures.
                rep_trig         : (boolean) if true saves and overwrites report.
                print_trig       : (boolean) if true prints sub_reports to console.
                display          : (boolean) if true displays figs (not recommended)
                
    Returns:    report     : (str) contains all error information
    '''
    tic_model = time() # Start model performance timer

    x_train, y_train, x_test, y_test = data # Decompose tuple into datasets

    # Initialize total error tracks to report at end of function
    tcoords_error = list()
    tbuild_missclass = tfloor_missclass = tstandard_error = 0
    subreports = list()    

    model_regressor.fit(x_train, y_train[QUANTITATIVE_COLUMNS])
    model_classifier.fit(x_train, y_train[CATEGORICAL_COLUMNS])
    
    
    # Loop through each phone_id, group them, compute and report errors.
    for phone_id, y_test_group in y_test.groupby("PHONEID", sort=False):
        
        # Obtain respective data for the phone_id labels (y_test_group)
        x_test_group = x_test.iloc[y_test_group.index]
        
        prediction_regressor = model_regressor.predict(x_test_group)
        prediction_classifier = model_classifier.predict(x_test_group)
        prediction = hstack((prediction_regressor, prediction_classifier))
        prediction = DataFrame(prediction, columns=QUANTITATIVE_COLUMNS+CATEGORICAL_COLUMNS)
        
        errors = compute_errors(prediction, y_test_group)
                
        # Format the errors into a string with headers for each error type.
        # Printing can be disabled due to noise. Append for final report.
        subreport = create_subreport(errors, y_test_group.shape[0], phone_id)
        if print_trig:
            print(subreport + '\n')
        subreports.append(subreport)
        
        # Take the phone_id errors and combine them with the running totals.
        build_missclass, floor_missclass, coords_error, standard_error = errors
        tcoords_error.extend(coords_error)
        tbuild_missclass += build_missclass
        tfloor_missclass += floor_missclass
        tstandard_error += standard_error
        
        # Plot pos vs. time and lat vs long. Save plots if boolean permits
        fig = plot_pos_vs_time(prediction, y_test_group, phone_id, model_name)
        if fig_trig:
            save_fig(fig, model_name, phone_id, "Position_vs_Timestamp")
        if not display:
            close(fig)
        
        fig = plot_lat_vs_lon(prediction, y_test_group, phone_id, model_name)
        if fig_trig:
            save_fig(fig, model_name, phone_id, "Latitude_vs_Longitude")    
        if not display:
            close(fig)
            
    # Compute totals report and print it
    terrors = (tbuild_missclass,tfloor_missclass,tcoords_error,tstandard_error)
    totals_report = create_subreport(terrors, y_test.shape[0])
    print(totals_report + '\n')
    
    toc_model = time()
    model_timer = toc_model - tic_model
    print("%s Timer: %.2f seconds" % (model_name, model_timer))
    
    # Create the output txt file of the entire report. Save if boolean permits.
    header = "%s\nModel Timer: %.2f seconds" % (model_name, model_timer)
    sub_reports = "\n\n".join(subreports)
    report = "\n\n".join([header, totals_report, sub_reports])
    if rep_trig:
        save_report(model_name, report, "phone_ids")
    
    return report
    

def run_model(model_name, model_regressor, model_classifier, data, rep_trig=SAVE_REPORT):
    '''
    This function will run the model in its entirety to evaluate preformance.
    It is ~6 seconds faster than the other function, so it is useful for
    querying for optimal parameters.
    
    Parameters: model_name       : (str)
                model_regressor  : (*) must have .fit and .predict class functions.
                model_classifier : (*) must have .fit and .predict class functions.
                data             : (tuple) contains the 4 sets of data.
                rep_trig         : (boolean) if true saves and overwrites report.
                
    Returns:    report           : (str) contains all error information
                prediction       : (DataFrame) the predicted output for each sample.
    '''
    tic_model = time() # Start model performance timer

    x_train, y_train, x_test, y_test = data # Decompose tuple into datasets

    model_regressor.fit(x_train, y_train[QUANTITATIVE_COLUMNS])
    model_classifier.fit(x_train, y_train[CATEGORICAL_COLUMNS])
    
    prediction_regressor = model_regressor.predict(x_test)
    prediction_classifier = model_classifier.predict(x_test)
    prediction = hstack((prediction_regressor, prediction_classifier))
    prediction = DataFrame(prediction, columns=QUANTITATIVE_COLUMNS+CATEGORICAL_COLUMNS)
    
    errors = compute_errors(prediction, y_test)
    
    # Compute totals report and print it
    totals_report = create_subreport(errors, y_test.shape[0])
    print(totals_report + '\n')
    
    toc_model = time()
    model_timer = toc_model - tic_model
    print("%s Timer: %.2f seconds" % (model_name, model_timer))
    
    # Create the output txt file of the entire report. Save if boolean permits.
    header = "%s\nModel Timer: %.2f seconds" % (model_name, model_timer)
    report = "\n\n".join([header, totals_report])
    if rep_trig:
        save_report(model_name, report, "totals")
    
    return report, prediction


################################## MAIN #######################################
    

if __name__ == "__main__":

    tic = time() # Start program performance timer
    
    close("all") # Close all previously opened plots
    
    if DISPLAY_PLOTS:
        ion()
    else:
        ioff()

    data = load_data("trainingData.csv", "validationData.csv", N, DROP_COLUMNS)    
        
    ################## INSERT MODEL AND MODEL NAME HERE #######################
    
    model_name = "KNN"
    model_regressor = KNeighborsRegressor(n_neighbors=3)
    model_classifier = KNeighborsClassifier(n_neighbors=1)
    
    #model_name = "DecisionTree"
    #model_regressor  = DecisionTreeRegressor()
    #model_classifier = DecisionTreeClassifier()
    
    #model_name = "RandomForest"
    #model_regressor = RandomForestRegressor(n_estimators=100)
    #model_classifier = RandomForestClassifier(n_estimators=100)

    #model_name = "NeuralNetworkRegressor"
    #model_regressor  = NeuralNetworkRegressor()
    #model_classifier = DecisionTreeClassifier() #dummy
    
    ################# INSERT MODEL AND MODEL NAME HERE ########################

    report, prediction = run_model(model_name, model_regressor, model_classifier, data)
    prediction = run_model_phone_id(model_name, model_regressor, model_classifier, data)
    
    toc = time() # Report program performance timer
    print("Program Timer: %.2f seconds" % (toc-tic))
