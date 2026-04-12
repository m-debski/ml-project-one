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
print("Continuous Feature Stats:")
def print_continuous_feature_stats(model: BinaryNaiveBayesModel):
    continuous_feature_stats = model.get_continuous_feature_stats()
    for class_label, feature_stats in continuous_feature_stats.items():
        print(f"    Stats for {class_label}:")
        for feature, stats in feature_stats.items():
            print(f"        Feature: {feature}")
            print(f"            mean:{feature_stats[feature]["mean"]}")
            print(f"            variance:{feature_stats[feature]["var"]}")
        print("\n")
print_continuous_feature_stats(model)

#3.
print("Top Predictive Features:")
def print_top_predictive_features(model: BinaryNaiveBayesModel) -> None:
    #TODO: am i meant to flip these for >50k or something????
    top_predictive_categories = model.get_top_predictive_categories(num=3)
    for class_label, predictive_stats in top_predictive_categories.items():
        print(f"    Features for {class_label}:")
        for feature, value, confidence in predictive_stats:
            print(f"        Value: {value} for Feature: {feature} with Confidence: {confidence}")
        print("\n")
print_top_predictive_features(model)

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

#1.
print("Model Evaluation:")
def print_evaluate_model(model: BinaryNaiveBayesModel, results_df: pd.DataFrame):
    evaluation_stats = model.evaluate(results_df)
    print(f"    Accuracy:{evaluation_stats["accuracy"]} \n")
    for class_label, metrics in evaluation_stats["per_class"].items():
        print(f"    Stats For {class_label}:")
        print(f"        Precision: {metrics["precision"]}")
        print(f"        Recall: {metrics["recall"]}")
        print(f"        F1 Score: {metrics["f1"]}")

    print(f"\n Confusion Matrix:\n")
    print(evaluation_stats["confusion_matrix"])
    print("\n")
print_evaluate_model(model, test_results)

#2. 
unseen_indicator = model.get_unseen_indicator()
print(f"Number of instances with unseen values in categorical fields: {len(test_results[unseen_indicator])}")
print("\n")

#3.
#TODO: again all confidence here is 0
print("Instances classified as >50k with high confidence")
df_3a = model.get_high_confidence_instances(test_results, 3, ">50K")
print("\n")
print(df_3a)
print("\n")
print("Instances classified as <=50K with high confidence")
df_3b = model.get_high_confidence_instances(test_results, 3, "<=50K")
print("\n")
print(df_3b)
print("\n")
print("Instances near the decision boundary (R ~1)")
df_3c = model.get_near_decision_boundary_instances(test_results, 3)
print("\n")
print(df_3c)
print("\n")


"""
Question 3:
1. Assign prediction labels to the unlabeled data set
2. Retrain your model on the combined supervised dataset and the pseudo-labelled dataset
3. Consider these extensions to improve the basic approach:
    a. Is it better to use unlabelled data, or only labelled with high confidence by the Q1 model?
    b. Does iterative label propagation help? FOr example label 50% of the unlabeled data with the q1 model, retrain
    then use the new model to label the remaining 50%, and train a final model on all available data
"""

#.1
unlabeled_df = pd.read_csv(FILE_UNLABELED, **CSV_READ_KWARGS)
unlabeled_df = unlabeled_df.dropna()
unlabeled_results = model.test(unlabeled_df)

model_labeled_df = unlabeled_results.copy()
#drop the has_unseen and confidence columns made in testing
model_labeled_df = model_labeled_df.drop(columns=[model.get_confidence_indicator(),model.get_unseen_indicator()])
#rename the predicted class label to the actual class label
model_labeled_df = model_labeled_df.rename(columns={model.get_prediction_indicator():CLASS_LABEL_COLUMN})

#2.
new_model = BinaryNaiveBayesModel(
    class_label_column=CLASS_LABEL_COLUMN,
    class_label_values=CLASS_LABEL_VALUES,
    categorical_feature_names=CATEGORICAL_FEATURE_NAMES,
    continuous_feature_names=CONTINUOUS_FEATURE_NAMES,
)
combined_df = pd.concat([training_df, model_labeled_df], axis=0, ignore_index=True)
new_model.train(combined_df)

#3. 
#TODO: need to do all this, it looks like quite a lot of permutations of models, but also a lot of like needing to find certain stats and stuff to evaluate with, hopefully its not too bad....


"""
Question 4:
1. Compare evaluation stats from Q1 and Q3 models
2. Compare the distribution of confidence between the models
3. Compare the estimated means/variances of continuous features and category probabilities of categorical features
4. Have the top predictors for each class changed? 
"""

#1.
new_test_results = new_model.test(test_df)
print("Q1 Model Evaluation:")
print_evaluate_model(model, test_results)
print("Q3 Model Evaluation:")
print_evaluate_model(new_model, new_test_results)

#2.

def print_confidence_distribution_data(results_df: pd.DataFrame, model: BinaryNaiveBayesModel) -> None:
    #TODO: actually do something with this
    confidence_column = model.get_confidence_indicator()
    confidence = results_df[confidence_column]

print("Confidence Distribution for Q1 Model:")
print_confidence_distribution_data(test_results, model)
print("Confidence Distribution for 3 Model:")
print_confidence_distribution_data(new_test_results, new_model)

#3.
print("Continuous Feature Stats Q1:")
print_continuous_feature_stats(model)


def print_categorical_feature_stats(model: BinaryNaiveBayesModel):
    categorical_feature_stats = model.get_categorical_feature_stats()
    for class_label, feature_stats in categorical_feature_stats.items():
        print(f"    Stats for {class_label}:")
        for feature, stats in feature_stats.items():
            print(f"        Feature: {feature}")
            for value, count in stats.items():
                print(f"            Value: {value} count: {count}")
        print("\n")
    

print("Categorical Feature Stats Q1:")
print_categorical_feature_stats(model)

print("Continuous Feature Stats Q3:")
print_continuous_feature_stats(new_model)

print("Categorical Feature Stats Q3:")
print_categorical_feature_stats(new_model)


#4.
print("Top Predictive Features for Model Q1")
print_top_predictive_features(model)
print("Top Predictive Features for Model Q3")
print_top_predictive_features(new_model)