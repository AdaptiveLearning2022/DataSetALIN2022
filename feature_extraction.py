import numpy as np
import pandas as pd


class Extraction:
    def __init__(self, first_data, behavior_data):
        self.first_data = first_data.copy()
        self.behavior_data = behavior_data.copy()

    def extract_test_features(self):
        self.first_data['numRight'] = np.round(
            self.first_data['testNumProblems'] * self.first_data['testPercentCorrect'])
        self.first_data['numMissed'] = self.first_data['testNumProblems'] - self.first_data['numRight']

        grp_sum = self.first_data.groupby(['studentID', 'sequenceID'])[
            ['testNumProblems', 'numRight', 'testTimespent']].sum()
        grp_nunique = self.first_data.groupby(['studentID', 'sequenceID'])[['topicID']].nunique()
        grp_nunique.reset_index(drop=False, inplace=True)
        grp_sum.reset_index(drop=False, inplace=True)

        merged = grp_sum.merge(grp_nunique, on=['studentID', 'sequenceID'])
        merged['testTopicCnt'] = merged['topicID']
        merged['testAvgCorrectRate'] = merged['numRight'] / merged['testNumProblems']
        merged['testProbAvgTime'] = np.log2(merged['testTimespent'] / merged['testNumProblems'] + 1)

        cols = ['studentID', 'sequenceID', 'testTopicCnt', 'testProbAvgTime', 'testAvgCorrectRate']
        merged = merged[cols]

        return merged

    def extract_behavior_features(self):
        behavior_df = self.behavior_data[pd.notna(self.behavior_data['behDateCompleted'])]
        grp_sum = self.behavior_data.groupby(['studentID', 'sequenceID'])[
            ['behNumProblems', 'behNumRight', 'behNumMissed', 'behNumSkipped', 'behTimeSpent']].sum()
        grp_cnt = self.behavior_data.groupby(['studentID', 'sequenceID'])[['behDateCompleted']].count()
        grp_nunique = behavior_df.groupby(['studentID', 'sequenceID'])[['topicID']].nunique()

        grp_sum.reset_index(drop=False, inplace=True)
        grp_cnt.reset_index(drop=False, inplace=True)
        grp_nunique.reset_index(drop=False, inplace=True)
        merged = grp_sum.merge(grp_cnt, on=['studentID', 'sequenceID'], how='left')
        merged = merged.merge(grp_nunique, on=['studentID', 'sequenceID'], how='left')

        merged['topicCnt'] = merged['topicID']
        merged['sheetCnt'] = merged['behDateCompleted']
        merged['topicAvgSheet'] = merged['sheetCnt'] / merged['topicCnt']
        merged['topicAvgTime'] = np.log2(merged['behTimeSpent'] / merged['topicCnt'] + 1)
        merged['sheetAvgTime'] = np.log2(merged['behTimeSpent'] / merged['sheetCnt'] + 1)
        merged['problemDone'] = np.log2(merged['behNumProblems'] - merged['behNumSkipped'] + 1)
        merged['probAvgTime'] = np.log2(merged['behTimeSpent'] / merged['problemDone'] + 1)
        merged['avgCorrectRate'] = merged['behNumRight'] / merged['problemDone']
        merged['behNumMissed'] = np.log2(merged['behNumMissed'] + 1)
        merged['behNumRight'] = np.log2(merged['behNumRight'] + 1)

        cols = ['studentID', 'sequenceID', 'topicCnt', 'sheetCnt', 'topicAvgSheet', 'topicAvgTime', 'sheetAvgTime',
                'problemDone', 'behNumRight', 'behNumMissed', 'probAvgTime', 'avgCorrectRate']
        merged = merged[cols]

        return merged

    def normalize_feature_value(self, row, feature_name):
        sequenceID = row['sequenceID']
        feature_value = row[feature_name]
        feature_min = self.grp_stat.loc[sequenceID, (feature_name, 'min')]
        feature_max = self.grp_stat.loc[sequenceID, (feature_name, 'max')]
        if feature_min < feature_max:
            feature_norm = (feature_value - feature_min) / (feature_max - feature_min)
        else:
            feature_norm = 1

        return feature_norm

    def normalize_features(self, combined):
        normalized = combined.copy()
        self.grp_stat = normalized.groupby(['sequenceID']).describe()

        normalized['normTestTopicCnt'] = normalized.apply(lambda row: self.normalize_feature_value(row, 'testTopicCnt'),
                                                          axis=1)
        normalized['normTestProbAvgTime'] = normalized.apply(
            lambda row: self.normalize_feature_value(row, 'testProbAvgTime'), axis=1)
        normalized['normTestAvgCorrectRate'] = normalized.apply(
            lambda row: self.normalize_feature_value(row, 'testAvgCorrectRate'), axis=1)
        normalized['normTopicDiff'] = normalized.apply(lambda row: self.normalize_feature_value(row, 'topicDiff'),
                                                       axis=1)
        normalized['normProbAvgTimeDiff'] = normalized.apply(
            lambda row: self.normalize_feature_value(row, 'probAvgTimeDiff'), axis=1)
        normalized['normCorrectRateDiff'] = normalized.apply(
            lambda row: self.normalize_feature_value(row, 'correctRateDiff'), axis=1)
        normalized['normTopicCnt'] = normalized.apply(lambda row: self.normalize_feature_value(row, 'topicCnt'), axis=1)
        normalized['normSheetCnt'] = normalized.apply(lambda row: self.normalize_feature_value(row, 'sheetCnt'), axis=1)
        normalized['normTopicAvgSheet'] = normalized.apply(
            lambda row: self.normalize_feature_value(row, 'topicAvgSheet'),
            axis=1)
        normalized['normTopicAvgTime'] = normalized.apply(lambda row: self.normalize_feature_value(row, 'topicAvgTime'),
                                                          axis=1)
        normalized['normSheetAvgTime'] = normalized.apply(lambda row: self.normalize_feature_value(row, 'sheetAvgTime'),
                                                          axis=1)
        normalized['normProblemDone'] = normalized.apply(lambda row: self.normalize_feature_value(row, 'problemDone'),
                                                         axis=1)
        normalized['normNumRight'] = normalized.apply(lambda row: self.normalize_feature_value(row, 'behNumRight'),
                                                      axis=1)
        normalized['normNumMissed'] = normalized.apply(lambda row: self.normalize_feature_value(row, 'behNumMissed'),
                                                       axis=1)
        normalized['normProbAvgTime'] = normalized.apply(
            lambda row: self.normalize_feature_value(row, 'probAvgTime'), axis=1)
        normalized['normAvgCorrectRate'] = normalized.apply(
            lambda row: self.normalize_feature_value(row, 'avgCorrectRate'), axis=1)

        cols = ['studentID', 'sequenceID', 'normTestTopicCnt', 'normTestProbAvgTime', 'normTestAvgCorrectRate',
                'normTopicDiff', 'normProbAvgTimeDiff', 'normCorrectRateDiff', 'normTopicCnt', 'normSheetCnt',
                'normTopicAvgSheet', 'normTopicAvgTime', 'normSheetAvgTime', 'normProblemDone', 'normNumRight',
                'normNumMissed', 'normProbAvgTime', 'normAvgCorrectRate', 'withBehavior']
        normalized = normalized[cols]

        return normalized

    def extract_and_normalize_features(self):
        test_features = self.extract_test_features()
        behavior_features = self.extract_behavior_features()
        combined = test_features.merge(behavior_features, on=['studentID', 'sequenceID'])
        combined['topicDiff'] = combined['topicCnt'] - combined['testTopicCnt']
        combined['probAvgTimeDiff'] = combined['probAvgTime'] - combined['testProbAvgTime']
        combined['correctRateDiff'] = combined['avgCorrectRate'] - combined['testAvgCorrectRate']
        combined['withBehavior'] = combined['sheetCnt'].apply(lambda cnt: True if cnt > 0 else False)
        combined.fillna(0, inplace=True)
        normalized = self.normalize_features(combined)

        return normalized
