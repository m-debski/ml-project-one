from BinaryNaiveBayesModel import BinaryNaiveBayesModel
import pandas as pd

USING_SMALLER_FILES = False
FILE_SUPERVISED_TRAIN = "data/adult_supervised_train.csv"
FILE_TEST = "data/adult_test.csv"
FILE_UNLABELED = "data/adult_unlabelled.csv"
if(USING_SMALLER_FILES):
    FILE_SUPERVISED_TRAIN = "data/stripped-adult_supervised_train.csv"
    FILE_TEST = "data/stripped-adult_test.csv"
    FILE_UNLABELED = "data/stripped-adult_unlabelled.csv"

CSV_READ_KWARGS = {"na_values": ["?"]}


CLASS_LABEL_COLUMN = "income"
CLASS_LABEL_VALUES = ["<=50K", ">50K"]
CATEGORICAL_FEATURE_NAMES = ["workclass","education","marital-status","occupation","relationship","race","sex",
"native-country"]
CONTINUOUS_FEATURE_NAMES = ["age","education-num","capital-gain","capital-loss","hours-per-week"]

model = BinaryNaiveBayesModel(
    class_label_column=CLASS_LABEL_COLUMN,
    class_label_values=CLASS_LABEL_VALUES,
    categorical_feature_names=CATEGORICAL_FEATURE_NAMES,
    continuous_feature_names=CONTINUOUS_FEATURE_NAMES,
)

training_df = pd.read_csv(FILE_SUPERVISED_TRAIN, **CSV_READ_KWARGS)
training_df = training_df.dropna()

model.train(training_df)

