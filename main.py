from re import S
from BinaryNaiveBayesModel import BinaryNaiveBayesModel
import pandas as pd

USING_SMALLER_FILES = True
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

"""
Question 1:
1. What are the prior probabilities of the two classes P(C)....
2. For each continuous feature, report mean and std dev
3. For each categorical feature, list the five most predictive category values for each class across all categorical features and their R values
"""

training_df = pd.read_csv(FILE_SUPERVISED_TRAIN, **CSV_READ_KWARGS)
training_df = training_df.dropna()
model.train(training_df)

#1. 
prior_probabilities = model.get_prior_probabilities()
print("Prior Probabilities:")
for class_label, proability in prior_probabilities.items():
    print(f"    P({class_label}) = {proability}")
print("\n")

#2.
continuous_feature_stats = model.get_continuous_feature_stats()
print("Continuous Feature Stats:")
for class_label, feature_stats in continuous_feature_stats.items():
    print(f"    Stats for {class_label}:")
    for feature, stats in feature_stats.items():
        print(f"        Feature: {feature}")
        print(f"            mean:{feature_stats[feature]["mean"]}")
        print(f"            variance:{feature_stats[feature]["var"]}")
    print("\n")

#3.
#TODO: am i meant to flip these for >50k or something????
top_predictive_categories = model.get_top_predictive_categories(num=3)
print("Top Predictive Features:")
for class_label, predictive_stats in top_predictive_categories.items():
    print(f"    Features for {class_label}:")
    for feature, value, confidence in predictive_stats:
        print(f"Value: {value} for Feature: {feature} with Confidence: {confidence}")
    print("\n")

"""
Question 2:
1. Report evaluation stats
2. Report how many test instances contained at least one unseen category value
3. Provide confidence:
    a. Instances classified as >50k with high confidence
    b. Instances classified as <=50K with high confidence
    c. Instances near the decision boundary (R ~1)
"""

test_df = pd.read_csv(FILE_TEST, **CSV_READ_KWARGS)
test_df = test_df.dropna()
test_results = model.test(test_df)


