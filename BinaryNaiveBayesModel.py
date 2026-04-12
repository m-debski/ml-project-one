from ClassModel import ClassModel
from pandas import DataFrame, Series
from math import log, exp

HAS_UNSEEN_SIGNIFIER = "_HAS_UNSEEN"
CLASS_LABEL_SIGNIFIER = "_CLASS_LABEL"
CONFIDENCE_SIGNIFIER = "_CONFIDENCE"

class BinaryNaiveBayesModel():
    def __init__(self, class_label_column: str, class_label_values: [], categorical_feature_names: [], continuous_feature_names: []) -> None:
        self._class_label_column = class_label_column
        self._class_label_values = tuple(class_label_values)
        self._categorical_feature_names = categorical_feature_names
        self._continuous_feature_names = continuous_feature_names

        self._class_models = {class_label: ClassModel() for class_label in class_label_values}

        self._categorical_unique_values = {feature: set() for feature in categorical_feature_names}


    def train(self, training_data: DataFrame) -> None:

        training_data = training_data.copy()

        #For each class label, fill in the counts for categorical features and the mean and var for continuous ones
        for class_label, model in self._class_models.items():
            model.train_class_model(
                train_data_partition = training_data[training_data[self._class_label_column] == class_label],
                train_data_len = len(training_data),
                categorical_feature_names = self._categorical_feature_names,
                continuous_feature_names = self._continuous_feature_names,
            )

        #Store the set of values seen for each categorical feature
        #TODO: there is definitely a more efficient way of doing this.....
        #TODO: is the term 'model' misleading here?
        for class_label, model in self._class_models.items():
            model_find =  model.get_unique_categorical_values(self._categorical_feature_names)

            self._categorical_unique_values = {
                key: self._categorical_unique_values.get(key, set()) |model_find.get(key, set())
                for key in self._categorical_unique_values.keys() | model_find.keys()
            }


    def test(self, testing_data: DataFrame) -> DataFrame:
        testing_data = testing_data.copy()

        #apply the predictive function to each instance
        test_results = testing_data.apply(self._predict_instance, axis = 1)
        testing_data["HAS_UNSEEN_SIGNIFIER"] = test_results.apply(lambda x: x["class_label"])
        testing_data["CLASS_LABEL_SIGNIFIER"] = test_results.apply(lambda x: x["contains_unseen_values"])
        testing_data[f"CONFIDENCE_SIGNIFIER"] = test_results.apply(lambda x: x["confidence"])

        return testing_data

    def _predict_instance(self, instance: Series) -> {}:
        #For each class label, calculate log(P(c|x))
        class_label_scores = {}
        contains_unseen_values = False

        for class_label, model in self._class_models.items():
            log_probability = log(model.get_prior_probability())
            
            for feature in self._categorical_feature_names:
                #Note the existence of unseen values for any categorical features
                if instance[feature] not in self._categorical_unique_values[feature]:
                    contains_unseen_values = True

                log_probability += model.compute_categorical_log_prob(
                    value=instance[feature],
                    feature_name=feature,
                    num_distinct_feature_values=len(self._categorical_unique_values[feature]),
                )

            for feature in self._continuous_feature_names:
                log_probability += model.compute_continuous_log_prob(
                    value=instance[feature],
                    feature_name=feature
                )

            class_label_scores[class_label] = log_probability

        c1, c2 = self._class_label_values
        #TODO: is this okay for confidence, effectively un-logging them?????
        confidence = exp(class_label_scores[c1] - class_label_scores[c2])

        return {
            "class_label": max(class_label_scores, key=class_label_scores.get),
            "contains_unseen_values": contains_unseen_values,
            "confidence": confidence,
        }