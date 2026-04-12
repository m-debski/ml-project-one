from ClassModel import ClassModel
import pandas as pd
from pandas import DataFrame, Series
from math import log, exp

# Columns added by test(); kept stable for evaluate() and assignment reporting helpers.
COL_PREDICTED = "PREDICTED_INCOME"
COL_HAS_UNSEEN = "HAS_UNSEEN"
COL_CONFIDENCE = "CONFIDENCE"

class BinaryNaiveBayesModel():
    def __init__(self, class_label_column: str, class_label_values: [], categorical_feature_names: [], continuous_feature_names: []) -> None:
        self._class_label_column = class_label_column
        self._categorical_feature_names = categorical_feature_names
        self._continuous_feature_names = continuous_feature_names

        self._class_label_values = tuple(class_label_values)
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
        #TODO: this is meant to be unique values for everyone right?
        for feature in self._categorical_feature_names:
            self._categorical_unique_values[feature] = set(
                training_data[feature].dropna().unique().tolist()
            )


    def test(self, testing_data: DataFrame) -> DataFrame:
        testing_data = testing_data.copy()

        #apply the predictive function to each instance
        test_results = testing_data.apply(self._predict_instance, axis=1)
        testing_data[COL_PREDICTED] = test_results.apply(lambda x: x["class_label"])
        testing_data[COL_HAS_UNSEEN] = test_results.apply(lambda x: x["contains_unseen_values"])
        testing_data[COL_CONFIDENCE] = test_results.apply(lambda x: x["confidence"])

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

    def evaluate(self, test_results: DataFrame) -> {}:
        actual = test_results[self._class_label_column]
        predicted = test_results[COL_PREDICTED]

        total_tp = 0
        total_tn = 0
        total_fp = 0
        total_fn = 0 
        per_class = {}

        for class_label in self._class_label_values:
            tp = int(((actual == class_label) & (predicted == class_label)).sum())
            tn = int(((actual != class_label) & (predicted != class_label)).sum())
            fp = int(((actual != class_label) & (predicted == class_label)).sum())
            fn = int(((actual == class_label) & (predicted != class_label)).sum())

            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn

            precision = 0.0
            if(tp + fp):
                precision = tp / (tp + fp)

            recall = 0.0
            if (tp + fn):
                recall = tp / (tp + fn)

            f1 = 0.0
            if (precision + recall):
                f1 = (2 * precision * recall / (precision + recall))

            per_class[class_label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)

        confusion_matrix = pd.crosstab(
            actual,
            predicted,
            rownames=["Actual"],
            colnames=["Predicted"],
        )

        return {
            "accuracy": accuracy,
            "confusion_matrix": confusion_matrix,
            "per_class": per_class,
        }

    #METHODS REQUIRED JUST TO ANSWER QUESTIONS

    def get_prior_probabilities(self) -> {}:
        priors = {}
        for class_label, model in self._class_models.items():
            priors[class_label] = model.get_prior_probability()
        return priors
            

    def get_continuous_feature_stats(self) -> {}:
        stats = {}
        for class_label, model in self._class_models.items():
            stats[class_label] = model.get_continuos_feature_stats()
        return stats

    def get_categorical_feature_stats(self) -> {}:
        stats = {}
        for class_label, model in self._class_models.items():
            stats[class_label] = model.get_categorical_feature_stats()
        return stats


    def get_top_predictive_categories(self, num: int) -> dict:
        #TODO: comment this better
        class_label_1 = self._class_label_values[0]
        class_label_2 = self._class_label_values[1]
        model_1 = self._class_models[class_label_1]
        model_2 = self._class_models[class_label_2]

        results = []
        for feature in self._categorical_feature_names:

            num_unique_values = len(self._categorical_unique_values[feature])
            all_values = set(model_1.get_categorical_feature_stats()[feature].keys()) | set(
                model_2.get_categorical_feature_stats()[feature].keys()
            )

            for value in all_values:
                #apply la place smoothing
                count1 = model_1.get_categorical_feature_stats()[feature].get(value, 0)
                count2 = model_2.get_categorical_feature_stats()[feature].get(value, 0)
                p1 = (count1 + 1) / (model_1.get_train_partition_len() + num_unique_values)
                p2 = (count2 + 1) / (model_2.get_train_partition_len() + num_unique_values)
                r_value = p2 / p1 
                results.append((feature, value, r_value))

        return {
            class_label_1: sorted(results, key=lambda x: x[2], reverse=True)[:num],
            class_label_2: sorted(results, key=lambda x: x[2])[:num]
        }

    #TODO: surely a better way to do this...?
    def get_unseen_indicator(self) -> str:
        return COL_HAS_UNSEEN

    def get_confidence_indicator(self) -> str:
        return COL_CONFIDENCE

    def get_prediction_indicator(self) -> str:
        return COL_PREDICTED

    def get_high_confidence_instances(self, test_results: DataFrame, count:int , classification: str) -> DataFrame:
        #TODO: is this ad hoc?
        classified = test_results[test_results[COL_PREDICTED] == classification]
        if classification == self._class_label_values[0]:
            classified = classified.sort_values(by=COL_CONFIDENCE, ascending=False)
        elif classification == self._class_label_values[1]:
            classified = classified.sort_values(by=COL_CONFIDENCE, ascending=True)
        return classified.head(count)

    def get_near_decision_boundary_instances(self,  test_results: DataFrame, count:int) -> DataFrame:
        #TODO: explain this better
        return test_results.iloc[(test_results[COL_CONFIDENCE] - 1).abs().argsort()].head(count)

