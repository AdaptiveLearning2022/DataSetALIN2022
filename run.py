import pandas as pd
from boosting import GBRT
from baseline import Baseline
from regression import Regression
from feature_extraction import Extraction
from evaluation import evaluate_point_prediction
from evaluation import evaluate_label_prediction

if __name__ == '__main__':
    test_data = pd.read_csv('data/student_test_data.csv', header=0)
    first_data = pd.read_csv('data/student_sequence_first_test.csv', header=0)
    point_data = pd.read_csv('data/student_sequence_test_points.csv', header=0)
    behavior_data = pd.read_csv('data/student_sequence_period_behavior.csv', header=0)

    # to extraction and normalize features
    extractor = Extraction(first_data, behavior_data)
    features = extractor.extract_and_normalize_features()

    # to create the original dataset
    dataset = features.merge(point_data, on=['studentID', 'sequenceID']).merge(test_data, on='studentID')
    without_behavior_data = dataset[dataset['withBehavior'] == False]
    with_behavior_data = dataset[dataset['withBehavior'] == True]

    baseline = Baseline()
    regression = Regression()
    gbrt = GBRT()

    # predicting last point of each sequence per student
    base_results = baseline.tran_and_predict(dataset)
    reg_results = regression.train_and_predict_separately(dataset)
    gbrt_results = gbrt.train_and_predict_separately(dataset)

    # merging results in order to make comparison
    results = base_results.merge(reg_results, on=['studentID', 'sequenceID', 'firstPoint', 'lastPoint']).merge(
        gbrt_results, on=['studentID', 'sequenceID', 'firstPoint', 'lastPoint'])

    # evaluating the prediction results
    point_eval = evaluate_point_prediction(results)
    label_eval = evaluate_label_prediction(results)
    print(point_eval)
    print(label_eval)
