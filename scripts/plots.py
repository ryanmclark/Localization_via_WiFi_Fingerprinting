#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 22:32:21 2019

@authors: Ryan Clark, Matt Hong, So Sosaki

File Description:
The plots file is used to handle all of the plotting. There is a file the plots
the Latitude and Longitude vs. the Timestamp between the prediction and truth.
There is also a plot that shows the latitude vs longitude for both the 
prediction and the truth.
"""

# Libraries 
from numpy import linspace
from matplotlib.pyplot import subplots

def plot_pos_vs_time(prediction, truth, phone_id, model_name):
    '''
    Plots the Latitude vs. Timestamp and the Longitude vs. Timestamp. This
    first centers the Lat/Long to the starting value on the truth Lat/Long.
    Since plotting may be turned off, the figure is returned so that it may be
    saved in the appropiate directory.
    
    Parameters: prediction : (DataFrame)
                truth      : (DataFrame)
                phone_id   : (int)
                model_name : (str)
                
    Returns     fig        : (Figure)
    '''
    
    fig, ax = subplots(ncols=2, figsize=(11,7), sharex="all")
    
    title = "%s\nPhone ID: %d" % (model_name, phone_id)    
    fig.suptitle(title)
    
    x = linspace(0, truth.shape[0]-1, truth.shape[0])
    for i, lat_long in enumerate(['LONGITUDE', 'LATITUDE']):
        y_predict = prediction[lat_long] - truth[lat_long].values[0]
        y_truth = truth[lat_long] - truth[lat_long].values[0]
        
        ax[i].plot(x, y_predict, label="Prediction")
        ax[i].plot(x, y_truth, label="Ground Truth")
        
        ax[i].set_title("%s vs. timestamp" % lat_long.lower())
        ax[i].set_ylabel("%s (meters)" % lat_long.lower())
        ax[i].set_xlabel("timestamp (number)")
        
        ax[i].legend()
        ax[i].grid()
        ax[i].set_xlim([min(x), max(x)])
        
    return fig

def plot_lat_vs_lon(prediction, truth, phone_id, model_name):
    '''
    Plots the Latitude vs. Longitude. Since plotting may be turned off, the 
    figure is returned so that it may be saved in the appropiate directory.
    
    Parameters: prediction : (DataFrame)
                truth      : (DataFrame)
                phone_id   : (int)
                model_name : (str)
                
    Returns     fig        : (Figure)
    '''
    fig, ax = subplots()
    
    title = "%s\nPhone ID: %d" % (model_name, phone_id)    
    fig.suptitle(title)
    
    ax.scatter(prediction["LONGITUDE"], prediction["LATITUDE"],
               label="Prediction")
    
    ax.scatter(truth["LONGITUDE"], truth["LATITUDE"], label="Ground Truth",s=7)
    
    ax.set_title("latitude vs. longitude")
    ax.set_ylabel("latitude (meters)")
    ax.set_xlabel("longitude (meters)")
    
    ax.legend()
    ax.grid()

    return fig
    