from pandas import DataFrame
import numpy as np

class ClassModel():
    def __init__(self):
        #TODO: is this a good way to init blank stuff for now?
        self._train_data_partition = DataFrame()
        self._train_partition_len = 0
        self._prior_probability = 0
        self._continuous_feature_stats = {}
        self._categorical_feature_stats = {}

    def train_class_model(self, train_data_partition: DataFrame, train_data_len: int, categorical_feature_names: [], continuous_feature_names: []):
        self._train_data_partition = train_data_partition
        self._train_partition_len = len(train_data_partition)
        self._prior_probability = self._train_partition_len / train_data_len

        #Populates the dictionary with features as keys, then dictionaries of {feature value: count} as the values
        for feature in categorical_feature_names:
            self._categorical_feature_stats[feature] = self._train_data_partition[feature].value_counts().to_dict()
        
        #Populates the dictionary with features as keys, then dictionaries of {"mean":mean and "var":var} as the values
        for feature in continuous_feature_names:
            self._continuous_feature_stats[feature] = {
                "mean": np.mean(self._train_data_partition[feature].tolist()),
                "var": np.var(self._train_data_partition[feature].tolist())
            }

    #TODO: this could surely be optimised....
    def get_unique_categorical_values(self, categorical_feature_names: []) -> dict:
        unique_values = {}
        for feature in categorical_feature_names:
            feature_unique_values = set(self._train_data_partition[feature].tolist())
            unique_values[feature] = feature_unique_values
        return unique_values

    

