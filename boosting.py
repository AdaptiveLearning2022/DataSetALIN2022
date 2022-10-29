import numpy as np
import pandas as pd
from evaluation import evaluate_rmse
from sklearn.ensemble import GradientBoostingRegressor as gbrt


class GBRT:
    def __init__(self):
        pass

    def train_model(self, train_X, train_y, params):
        model = gbrt(**params)
        model.fit(train_X, train_y)

        return model

    def predict(self, model: gbrt, test_X):
        pred = model.predict(test_X)
        pred = np.round(pred)
        pred = np.maximum(pred, 0)  # point should be non-negative number

        return pred

    def search_optimal_params_by_grid(self, train_X, train_y, valid_X, valid_y):
        min_rmse = 999
        opt_params = {}
        for n_estimators in np.arange(100, 1100, 100):
            for max_depth in np.arange(1, 11, 1):
                for learning_rate in np.arange(0.01, 0.11, 0.01):
                    params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'learning_rate': learning_rate,
                              'random_state': 0}
                    model = self.train_model(train_X, train_y, params)
                    pred = self.predict(model, valid_X)
                    rmse = evaluate_rmse(valid_y, pred)
                    if rmse < min_rmse:
                        min_rmse = rmse
                        opt_params = params

        return opt_params

    def train_and_predict(self, data, features, target):
        train = data[data['dataType'] == 'train'].copy()
        valid = data[data['dataType'] == 'valid'].copy()
        test = data[data['dataType'] == 'test'].copy()

        opt_params = self.search_optimal_params_by_grid(train[features], train[target], valid[features], valid[target])
        model = self.train_model(train[features], train[target], opt_params)
        pred = self.predict(model, test[features])
        test['gbrt'] = pred

        return test[['studentID', 'sequenceID', 'firstPoint', 'lastPoint', 'gbrt']]

    # to train one model and make predictions by the data containing behavior
    # then to train one other model and make predictions by the data containing no behavior
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
