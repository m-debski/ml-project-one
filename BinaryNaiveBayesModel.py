from ClassModel import ClassModel
from pandas import DataFrame

class BinaryNaiveBayesModel():
    def __init__(self, class_label_column: str, class_label_values: [], categorical_feature_names: [], continuous_feature_names: []) -> None:
        self._class_label_column = class_label_column
        self._categorical_feature_names = categorical_feature_names
        self._continuous_feature_names = continuous_feature_names

        self._class_models = {class_label: ClassModel() for class_label in class_label_values}

        self._categorical_unique_values = {feature: set() for feature in categorical_feature_names}


    def train(self, training_data: DataFrame):

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
        for class_label, model in self._class_models.items():
            model_find =  model.get_unique_categorical_values(self._categorical_feature_names)

            self._categorical_unique_values = {
                key: self._categorical_unique_values.get(key, set()) |model_find.get(key, set())
                for key in self._categorical_unique_values.keys() | model_find.keys()
            }


        
        