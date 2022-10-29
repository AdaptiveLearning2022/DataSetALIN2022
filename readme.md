# Student Test Results Prediction based on Learning Behavior: Learning Beyond Tests

Dataset Part A: The Goal is to predict Test Results, in the form of averaged correctness, averaged timespent in the test, based only on the learning history (learning  behavior records)

Dataset Part B: The objective is to predict the last test results, points and scores, based on the learning behavior records and the first test results.

# Authors

Guijia He, Chengwei Huang, Steven Yang, Kelvin Lwin, Ran Ju, Yuanmi Chen, Xiaoming Zhu.

# Institutions

Co-developed by ALIN.ai, and Zhejiang Lab  

2022-10-30

# The DataSet

The raw data is provided by ALIN.ai where a large number of students participated in math learning and tests, online. 
Dataset_A:
The feature constructed from the raw data is achieved by applying statistic functionals to the backend data sheets where learning behavior is recorded, such as 'points earned' in a 'learning session'. 
The final cohort we build from this learning senario consists of the predicting target: averaged correctness, and the averaged timespent, and the input features (43 dimensions). 
The input features consist of two parts, one contains information(5 dimensions) on the test itself, e.g. the difficulty; the other contains the rest 38 dimensions of features.  

Dataset_B:
The dataset contains the test results including the first and the last tests, as well as the behavior learning records between the two tests.
The grains of the dataset include test, sequence, topic and problem, from coarse to fine.
The features extracted from the dataset are based on the sequence grain, such as the number of problems of each sequence in the first test.
The target is to predict the point of each sequence and the total socre of the last test. 

# Baseline Results

Model_A (based on Dataset_A): The baseline predictor is build by XGBoostRegressor, with an accuracy above 80% was observed on the averaged correctness prediction, while an only 10% accuracy was observed on averaged timespent prediction.

LightGBM,RandomForest,and DNN models are also implemented for comparison.

Model_B (based on Dataset_B): The baseline assumed the points and socres of the last test are equal to those of the first test.

# Files Description
## student_data_processed.csv
contains data cohort used for modeling
## predictExperiment.py
contains python machine learning scripts for modeling
##student_test_data.csv
contains the scores of the first and last tests for each student, along with its data type (train/validation/test)
##student_sequence_test_points.csv
contains the points of each sequence of the first and last tests for each student
##student_sequence_first_test.csv
contains some sequence-related features of the first test for each student
##student_sequence_period_behavior.csv
contains some sequence-related features extracted from behavioral records during the first and the last tests for each student
##feature_extraction.py
extracts normalized features from the dataset
##baseline.py
yields the results of baseline model
##boosting.py
yields the results of GBRT model
##regression.py
yields the results of regression model
##evaluation.py
evaluates the performance of models by comparing with the groud truth
##run.py
sequentially executes the process
## readme.md
this file
