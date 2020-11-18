# imdb-wiki-assignment
Code-base developed for cleaning and processing IMDB-Wiki Image dataset 

DIRECTORY STRUCTURE OF REPOSITORY

code:
1. clean_data.py - Extracts, Compiles, Cleand the IMDB, Wiki data
2. plot_age_distribution.py - Plot the frequency distributions for age
3. model_training.py - Feature extraction from VGG-16 and age classification model training

utility:
1. extract_data.py - For extracting DOB, Date of photo taken, gender, filepath, face score 1 & 2 from the metadata for Wiki & IMDB datasets
2. frequency_bins.py - Used for plotting histogram by age-buckets (based on quantile cuts)
3. model_training_functions - Contains function definitions for VGG-16 Feature Extraction & Age Classification Model Architecture

plots: Contains the different frequency distributions of age for the combined IMDB-Wiki dataset

Files downloaded from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
