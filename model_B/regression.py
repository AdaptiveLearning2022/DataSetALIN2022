import numpy as np
import pandas as pd
from evaluation import evaluate_rmse
from sklearn.linear_model import LinearRegression


class Regression:
    def __init__(self):
        pass

    def train_model(self, train_X, train_y):
        model = LinearRegression()
        model.fit(train_X, train_y)

        return model

    def predict(self, model: LinearRegression, test_X):
        pred = model.predict(test_X)
        pred = np.round(pred)
        pred = np.maximum(pred, 0)  # point should be non-negative number

        return pred

    def extract_opt_features_by_greedy(self, train_X, train_y, valid_X, valid_y):
        candidate_features = train_X.columns.tolist()
        opt_features = []
        min_rmse = 999

        # adding only one feature each time that can reduce rmse most
        # until rmse cannot be reduced anymore
        while len(candidate_features) > 0:
            rmse_dict = {}
            for f in candidate_features:
                features = opt_features.copy()
                features.append(f)
                model = self.train_model(train_X[features], train_y)
                pred = self.predict(model, valid_X[features])
                rmse = evaluate_rmse(valid_y, pred)
                rmse_dict[f] = rmse

            opt_item = sorted(rmse_dict.items(), key=lambda item: item[1])[0]
            opt_f = opt_item[0]
            opt_rmse = opt_item[1]

            if opt_rmse < min_rmse:
                min_rmse = opt_rmse
                opt_features.append(opt_f)
                candidate_features.remove(opt_f)
            else:
                break

        return opt_features

    def train_optimal_model_by_valid(self, train_X, train_y, valid_X, valid_y):
        optiaml_features = self.extract_opt_features_by_greedy(train_X, train_y, valid_X, valid_y)
        model = self.train_model(train_X[optiaml_features], train_y)

        return model

    def train_and_predict(self, data, features, target):
        train = data[data['dataType'] == 'train'].copy()
        valid = data[data['dataType'] == 'valid'].copy()
        test = data[data['dataType'] == 'test'].copy()

        opt_features = self.extract_opt_features_by_greedy(train[features], train[target], valid[features],
                                                           valid[target])
        model = self.train_model(train[opt_features], train[target])
        pred = self.predict(model, test[opt_features])
        test['regression'] = pred

        return test[['studentID', 'sequenceID', 'firstPoint', 'lastPoint', 'regression']]

    # training one model and making predictions by the data containing learning behavior
    # then training one other model and making predictions by the data containing no behavior
    def train_and_predict_separately(self, dataset: pd.DataFrame):
        with_behavior_data = dataset[dataset['withBehavior'] == True]
        without_behavior_data = dataset[dataset['withBehavior'] == False]

        without_behavior_features = ['normTestTopicCnt', 'normTestProbAvgTime', 'normTestAvgCorrectRate', 'firstPoint']
        with_behavior_features = ['normTestTopicCnt', 'normTestProbAvgTime', 'normTestAvgCorrectRate', 'normTopicDiff',
                                  'normProbAvgTimeDiff', 'normCorrectRateDiff', 'normTopicCnt', 'normSheetCnt',
                                  'normTopicAvgSheet', 'normTopicAvgTime', 'normSheetAvgTime', 'normProblemDone',
                                  'normNumRight', 'normNumMissed', 'normProbAvgTime', 'normAvgCorrectRate',
                                  'firstPoint']
        label = 'lastPoint'

        without_behavior_pred = self.train_and_predict(data=without_behavior_data, features=without_behavior_features,
                                                       target=label)
        with_behavior_pred = self.train_and_predict(data=with_behavior_data, features=with_behavior_features,
                                                    target=label)
        results = pd.concat([without_behavior_pred, with_behavior_pred])

        return results
