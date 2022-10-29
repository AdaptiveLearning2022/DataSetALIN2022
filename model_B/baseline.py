import pandas as pd


class Baseline:
    def __init__(self):
        pass

    def tran_and_predict(self, data: pd.DataFrame):
        test = data[data['dataType'] == 'test'].copy()
        test['baseline'] = test['firstPoint']

        return test[['studentID', 'sequenceID', 'firstPoint', 'lastPoint', 'baseline']]
