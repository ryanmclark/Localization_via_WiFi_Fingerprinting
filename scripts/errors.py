#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 22:17:15 2019

@author: Ryan Clark

File Description:
The errors file is used to calculate the various types of errors that are used
in this dataset. Localization Error is the euclidean distance between each
latitude and longitude measurement for the prediction and truth. The Number of
Misclassified is the number of missclassified samples. Standard Error is the
error defined by the creators of the dataset for the IPIN2015 competition. In
this competition building missclassifications were penalized by 50 meters each 
and floor missclassifications were penatlized by 4 meters each. The Standard
Error is given by the pentalties multiplied by the number of missclassifcations
plus the Localization Error. The main function here is compute_errors while
everything else is a helper function.
"""

#Libraries
from numpy import sqrt, square, sum

# Hyper-parameters
BP = 50 # Default Building Penalty
FP = 4 # Default Floor Penalty

def localizaion_error(prediction, truth):
    '''
    Computes the Localization Error by computing the euclidean distance between
    the predicted latitude and longitude and the true latitude and longitude.
    
    Parameters: prediction : (Dataframe)
                truth      : (Dataframe)
            
    Returns:    error      : (array) error between each sample
    '''
    x, y  = prediction['LONGITUDE'].values, prediction['LATITUDE'].values
    x0, y0 = truth['LONGITUDE'].values, truth['LATITUDE'].values
    error = sqrt(square(x - x0) + square(y - y0))
    return error
    
def number_missclassified(prediction, truth, column_name):
    '''
    Computes the number of missclassifications by summing how many elements
    do not match between the prediction and truth columns. The column_name
    parameter is there because this can be used for the Floor or the Building.
    
    Parameters: prediction  : (Dataframe)
                truth       : (Dataframe)
                column_name : (str) specifies which column to compute the error
            
    Returns:    error       : (int) total number of missclassifications.
    '''
    error = sum(prediction[column_name].values != truth[column_name].values)
    return error
    
def compute_errors(prediction, truth, building_penalty=BP, floor_penalty=FP):
    '''
    Computes the missclassification errors, localization error, and standard
    error. For me detail, see the File Description.
    
    Parameters: prediction       : (Dataframe)
                truth            : (Dataframe)
                building_penalty : (int)
                floor_penalty    : (int)
            
    Returns:    errors           : (tuple) contains all error types
    '''
    build_missclass = number_missclassified(prediction, truth, "BUILDINGID")
    
    floor_missclass = number_missclassified(prediction, truth, "FLOOR")
    
    coords_error = localizaion_error(prediction, truth)
    
    standard_error = (building_penalty * build_missclass + floor_penalty *
                      floor_missclass + sum(coords_error))
    
    errors = (build_missclass, floor_missclass, coords_error, standard_error)
                         
    return errors