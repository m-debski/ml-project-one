from pandas import DataFrame
import numpy as np
from math import log, pi

LAPLACE = 1
#TODO: is this valid EPSILON????
EPSILON = 1e-9

class ClassModel():
    def __init__(self):
        self._train_data_partition = None
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

    def compute_categorical_log_prob(self, value: str, feature_name: str, num_distinct_feature_values: int) -> float:
        #get the count of this feature value, based on the features stats
        count = self._categorical_feature_stats[feature_name].get(value, 0)
        return log( (count + LAPLACE) / self._train_partition_len + LAPLACE * num_distinct_feature_values)

    def compute_continuous_log_prob(self, value: str, feature_name: str) -> float:
        mean = self._continuous_feature_stats[feature_name]["mean"]
        var = self._continuous_feature_stats[feature_name]["var"]
        if var == 0:
            var = EPSILON
        return -0.5 * log(2 * pi * var) - ((value - mean) ** 2) / (2 * var)

    def get_prior_probability(self) -> float:
        return self._prior_probability

    def get_train_partition_len(self) -> int:
        return self._train_partition_len

    #METHODS JUST FOR ANSWERING QUESTIONS

    def get_continuos_feature_stats(self) -> {}:
        return self._continuous_feature_stats
    
    def get_categorical_feature_stats(self) -> {}:
        return self._categorical_feature_stats

