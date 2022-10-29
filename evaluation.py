import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def evaluate_rmse(truth, pred):
    return np.sqrt(mean_squared_error(truth, pred))


def evaluate_point_prediction(results: pd.DataFrame):
    eval = pd.DataFrame([], columns=['MAE', 'RMSE', 'R2'], index=['baseline', 'regression', 'gbrt'])
    eval.loc['baseline', 'MAE'] = mean_absolute_error(results['lastPoint'], results['baseline'])
    eval.loc['baseline', 'RMSE'] = evaluate_rmse(results['lastPoint'], results['baseline'])
    eval.loc['baseline', 'R2'] = r2_score(results['lastPoint'], results['baseline'])
    eval.loc['regression', 'MAE'] = mean_absolute_error(results['lastPoint'], results['regression'])
    eval.loc['regression', 'RMSE'] = evaluate_rmse(results['lastPoint'], results['regression'])
    eval.loc['regression', 'R2'] = r2_score(results['lastPoint'], results['regression'])
    eval.loc['gbrt', 'MAE'] = mean_absolute_error(results['lastPoint'], results['gbrt'])
    eval.loc['gbrt', 'RMSE'] = evaluate_rmse(results['lastPoint'], results['gbrt'])
    eval.loc['gbrt', 'R2'] = r2_score(results['lastPoint'], results['gbrt'])

    return eval


def evaluate_label_prediction(results: pd.DataFrame):
    grp = results.groupby('studentID').sum()
    grp['label'] = grp.apply(lambda row: True if row['lastPoint'] >= row['firstPoint'] else False, axis=1)
    grp['base_label'] = grp.apply(lambda row: True if row['baseline'] >= row['firstPoint'] else False, axis=1)
    grp['reg_label'] = grp.apply(lambda row: True if row['regression'] >= row['firstPoint'] else False, axis=1)
    grp['gbrt_label'] = grp.apply(lambda row: True if row['gbrt'] >= row['firstPoint'] else False, axis=1)

    eval = pd.DataFrame([], columns=['Precision', 'Recall', 'F1-score'], index=['baseline', 'regression', 'gbrt'])
    eval.loc['baseline', 'Precision'] = precision_score(grp['label'], grp['base_label'], average='weighted')
    eval.loc['baseline', 'Recall'] = recall_score(grp['label'], grp['base_label'], average='weighted')
    eval.loc['baseline', 'F1-score'] = f1_score(grp['label'], grp['base_label'], average='weighted')
    eval.loc['regression', 'Precision'] = precision_score(grp['label'], grp['reg_label'], average='weighted')
    eval.loc['regression', 'Recall'] = recall_score(grp['label'], grp['reg_label'], average='weighted')
    eval.loc['regression', 'F1-score'] = f1_score(grp['label'], grp['reg_label'], average='weighted')
    eval.loc['gbrt', 'Precision'] = precision_score(grp['label'], grp['gbrt_label'], average='weighted')
    eval.loc['gbrt', 'Recall'] = recall_score(grp['label'], grp['gbrt_label'], average='weighted')
    eval.loc['gbrt', 'F1-score'] = f1_score(grp['label'], grp['gbrt_label'], average='weighted')

    return eval
