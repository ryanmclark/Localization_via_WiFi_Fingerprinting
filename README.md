# Localization_via_WiFi_Fingerprinting
Multi-Floor Indoor Localization based on Wi-Fi Fingerprinting using various supervised machine learning models on the UJIIndoorLoc dataset. This dataset covers a 110m^2 area at the Universitat Jaume I and can be used for classification among 3 buildings and 4 floors as well as regression for latitude and longitude measurements in meters. There are 21,048 samples in this dataset that contain 529 features. Of the 529 features, there are 520 WAPs with intensity values and 9 types of labels. The 9 types of labels include latitude, longitude, building ID, floor ID, space ID, relative position ID, user ID, phone ID, and timestamp. The classification and regression models used here are K-Nearest Neighbors, Random Forrest, Decision Tree, and Support Vector Machine. Feature selection is also done via Variance Thresholding or Principle Component Analysis. All models and feature selection are implemented through the sklearn Python Package.

For more information on the dataset, please visit:
https://archive.ics.uci.edu/ml/datasets/ujiindoorloc 

Dataset Link:
https://archive.ics.uci.edu/ml/machine-learning-databases/00310/

## Output Reports:
Building Error: sum of all missclassified building samples over the total sample count * 100 (percent)  
Floor Error: sum of all missclassified floor samples over the total sample count * 100 (percent)  
Mean Coordinate Error: mean euclidean error from estimated latitude and longitude against the true latitude and longitude.  
Standard Error: sum of BuildingPenalty*BuildingError + FloorPenalty*FloorError + Coordinate Error where the BuildingPenalty is 50 and the FloorPenalty is 4.  

Note that is possible to have a correct floor but an incorrect building.  

There are also plots for each phone id for latitude vs. timestamp, longitude vs. timestamp, and latitude vs. longitude for prediction against ground truth.  

## Current Results:  
K-Nearest Neighbors  
Model Timer: 29.35 seconds  
Mean Coordinate Error: 1.28 +/- 7.95 meters  
Standard Error: 5307.65 meters  
Building Percent Error: 0.21%  
Floor Percent Error: 0.24%  
Prob that Coordinate Error Less than 10m: 97.30%  

Random Forest Regressor  
Model Timer: 106.05 seconds  
Mean Coordinate Error: 4.09 +/- 8.52 meters  
Standard Error: 16242.91 meters  
Building Percent Error: 0.29%  
Floor Percent Error: 0.52%  
Prob that Coordinate Error Less than 10m: 92.72%  

Decision Tree  
Model Timer: 3.70 seconds  
Mean Coordinate Error: 4.65 +/- 11.55 meters  
Standard Error: 18969.19 meters  
Building Percent Error: 0.34%  
Floor Percent Error: 3.69%  
Prob that Coordinate Error Less than 10m: 87.36%  

Support Vector Machine  
Model Timer: 7.58 seconds  
Mean Coordinate Error: 45.03 +/- 24.09 meters  
Standard Error: 172698.51 meters  
Building Percent Error: 0.29%  
Floor Percent Error: 0.81%  
Prob that Coordinate Error Less than 10m: 3.61%  

## File Descriptions:
This is just a brief overview. A more detailed explination is provided in each of the file's descriptions in the header.

#### main.py
Runs the dataset through the models in the main function and saves the output to the output directory.

#### data/*
Contains the data for the provided trainingData.csv and validationData.csv/

#### scripts/*
All of the helper functions and models are located here.

##### scripts/models.py
Contains the loading scripts for K-Nearest Neighbor, Random Forest, Decision Tree, and Support Vector Machine classifiers and regressors. PCA and Variance Thresholding functions are also provided here as well. All models are from the sklearn Python Package.

##### scripts/errors.py
Functions for the four types of errror functions - Building Missclassification, Floor Missclassification, Localization Error, and Standard Error.

##### scripts/plots.py
Functions to plot latitude vs. longitude and position vs. timestamp for prediction against ground truth.

##### scripts.utils.py
Helper functions used to load and preprocess data, format the output, and save output.

#### output/*
All outputs for each model are in their own subdirectory within here.

#### analysis/*
In here are a few plots generated during the parameter estimation process. They provide a few insides on the prior probabilities, data integrity, average intensities, WAP prevelence, and the relationship between intensity and floor for a couple of WAP IDs. The scripts used to generate these are not provided due to being messing one off scripts.

## Requirements
Python Version==3.7.1  

matplotlib==2.2.3  
numpy==1.15.4  
pandas==0.23.4  
scikit-learn==0.19.2  

## Acknowledgements
The dataset was created by:

Joaquín Torres-Sospedra, Raul Montoliu, Adolfo Martínez-Usó, Tomar J. Arnau, Joan P. Avariento, Mauri Benedito-Bordonau, Joaquín Huerta, Yasmina Andreu, óscar Belmonte, Vicent Castelló, Irene Garcia-Martí, Diego Gargallo, Carlos Gonzalez, Nadal Francisco, Josep López, Ruben Martínez, Roberto Mediero, Javier Ortells, Nacho Piqueras, Ianisse Quizán, David Rambla, Luis E. Rodríguez, Eva Salvador Balaguer, Ana Sanchís, Carlos Serra, and Sergi Trilles.

