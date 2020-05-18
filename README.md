# Figure-8-Data-science-project
Disaster Response Pipeline-figure 8

## Table of Contents
1. Libraries
2. Project overview
3. File Descriptions
4. Running Instructions
5. Acknowledgements


## Libraires

Python 3 is required to run the anlaysis
Python libraries that are used/requires are as follows:
1. Numpy
2. Pandas
3. Pickle
4. re
5. word_tokenize from ntlk.tokenize
6. pipeline, Featureunion from pipleine
7. CountVectorizer, TfidfTransformer from feature_extraction.text
8. MultiouputClassifier
9. Classification_report
10. RandomForestClassifier
11. create_engine from sqlalchemy

## Project overview

The provided repository is an web app used by emergency workers during any disater events. It classifies messages into several related categroies and notifies the associated deparment or aid agencies. 
This app used machine learning alogrithm that trains and classfies the messages in to related categoires. 

## File Descriptions

Folder: ETL- Etract Transform Load
This folder consists of
1. diaster_categories
2. diaster_messages
3. DiasterResponse SQL database
4. process_data.py

1 and 2 are the raw input files, 3 is the SQL database where the final cleaned data is stored, and 4 is the python code that cleans the input data and outputs the cleaned data

Folder: Model
This folder has train_classifier.py, a python script file that takes the cleaned data from the SQL database and trains the data using machine learning algorithm and categorizes the messages.

Folder: app
This folder has all the files that are required for running the web app.

ETL_Pipeline_Preparationipynb - jypyter notebook for cleaning the data (mimics process_data.py)
ML_Pipeline_Preparation.ipynb - jypyter notebook for categorizing the messages (mimics train_classifier.py)

## Running Instructions

Running process.py
 - To run ETL pipeline that cleans data and stores in database
        `python ETL/process_data.py ETL/disaster_messages.csv ETL/disaster_categories.csv ETL/DisasterResponse.db`

Running trainer_classifier.py
- To run ML pipeline that trains classifier and saves
        `python Model/train_classifier.py ETL/DisasterResponse.db Model/classifier.pkl`

Runnig web app.
   
  -save the app folder in the current working directory
  - run this command python run.py`
  - Go to http://0.0.0.0:3001/

Acknowledgements

This poject is a part of Udacity Nano Degree. Code, templates, and data are provided for the purpose of the project source by Udacity from Figure Eight.
