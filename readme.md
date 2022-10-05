# Student Test Results Prediction based on Learning Behavior: Learning Beyond Tests

The Goal is to predict Test Results, in the form of averaged correctness, averaged timespent in the test, based only on the learning history (learning  behavior records)

# Authors

Guijia He, Chengwei Huang, Steven Yang, Kelvin Lwin, Ran Ju, Yuanmi Chen, Xiaoming Zhu.

# Institutions

Co-developed by ALIN.ai, and Zhejiang Lab  

2022-9-20

# The DataSet

The raw data is provided by ALIN.ai where a large number of students participated in math learning and tests, online. 
The feature constructed from the raw data is achieved by applying statistic functionals to the backend data sheets where learning behavior is recorded, such as 'points earned' in a 'learning session'. 
The final cohort we build from this learning senario consists of the predicting target: averaged correctness, and the averaged timespent, and the input features (43 dimensions). 
The input features consist of two parts, one contains information(5 dimensions) on the test itself, e.g. the difficulty; the other contains the rest 38 dimensions of features.  

# Baseline Results

The baseline predictor is build by XGBoostRegressor, with an accuracy above 80% was observed on the averaged correctness prediction, while an only 10% accuracy was observed on averaged timespent prediction.

LightGBM,RandomForest,and DNN models are also implemented for comparison.

# Files Description

## student_data_processed.csv
contains data cohort used for modeling
## predictExperiment.py
contains python machine learning scripts for modeling
## readme.md
this file
