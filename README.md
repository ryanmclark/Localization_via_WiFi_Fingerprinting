# Localization_via_WiFi_Fingerprinting
Multi-Floor Indoor Localization based on Wi-Fi Fingerprinting using various Machine Learning models on the UJIIndoorLoc dataset.

## File Descriptions:
This is just a brief overview. A more detailed explination is provided in each of the file's descriptions in the header.

#### main.py
Runs the dataset through the models in the main function and saves the output to the output directory

#### scripts/*
All of the helper functions and models are saved here.

##### scripts/errors.py
Computes the four types of errror functions - Building Missclassification, Floor Missclassification, Localization Error, and Standard Error.

##### scripts/plots.py
Functions to plot latitude vs. longitude and position vs. timestamp

##### scripts.utils.py
Helper functions used to load and preprocess data, format the output, and save output.

#### /output/*
Location of all of the output for each model ran through the main function.
