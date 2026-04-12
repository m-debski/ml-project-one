import pandas as pd
import numpy as np
import math

LAPLACE = 1
EPSILON = 1e-9

class NaiveBayesModel():
    def __init__(self, class_column, class_values, categorical_columns, continuous_columns, data_set):
        self.data_set = data_set
        self.class_column = class_column #hard coded title of column that is for classes
        #TODO: can prolly remove this, just use the class states idiot:
        self.class_values = class_values #hard coded values for the class column
        self.categorical_columns = categorical_columns #list of strs hard coded
        self.continuous_columns = continuous_columns #list of strs hard coded

        self.class_states = [] #list of class state objects

        self._init_class_states()


    def _init_class_states(self):
        for value in self.class_values:

            state = ClassState(
                value, 
                self.data_set[self.data_set[self.class_column] == value], 
                len(self.data_set)
            )
            print(f"created class state for: {state.label} with count {state.count} and prob {state.prob}")
            self.class_states.append(state)


    def train(self):
        #for each class state, then for each feature compute their stats based on type

        for class_state in self.class_states:

            data_set = class_state.data_set 

            for column in self.categorical_columns:
                class_state.categorical_counts[column] = data_set[column].value_counts().to_dict()

            for column in self.continuous_columns:
                class_state.continuous_stats[column] = {
                    "mean": np.mean(data_set[column].tolist()),
                    "variance": np.var(data_set[column].tolist()),
                }

        self.k_per_feature = {
            col: int(self.data_set[col].nunique()) + 1 for col in self.categorical_columns
        }

        self.seen_categories = {
            col: set(self.data_set[col].dropna().unique()) for col in self.categorical_columns
        }

    def test(self, test_df):

        results = test_df.apply(self._predict_row, axis=1)
        test_df["PREDICTED_INCOME"] = results.apply(lambda x: x["prediction"])
        test_df["HAS_UNSEEN"] = results.apply(lambda x: x["had_unseen"])
        test_df[f"CONFIDENCE"] = results.apply(lambda x: x["confidence"])

        actual_values = test_df["income"]
        predicted_values = test_df["PREDICTED_INCOME"]

        #TODO: gotta be a better class based way to do this or smth

        total_true_positive = 0
        total_true_negative = 0
        total_false_positive = 0
        total_false_negative = 0

        for state in self.class_states:
            label = state.label
            true_positive = ((actual_values == label) & (predicted_values == label)).sum()
            total_true_positive+= true_positive
            true_negative = ((actual_values != label) & (predicted_values != label)).sum()
            total_true_negative+= true_negative
            false_positive = ((actual_values != label) & (predicted_values == label)).sum()
            total_false_positive+= false_positive
            false_negative = ((actual_values == label) & (predicted_values != label)).sum()
            total_false_negative+= false_negative

            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1 = 2*(precision*recall) / (precision + recall)

            print(f"Class: {label}")
            print(f"  Precision: {precision}")
            print(f"  Recall: {recall}")
            print(f"  F1 Score: {f1}")
            print("\n")

        accuracy = (total_true_positive + total_true_negative) / (total_true_positive + total_true_negative + total_false_positive + total_false_negative)
        print(f"Accuracy: {accuracy}")

        #TODO: check this
        confusion_matrix = pd.crosstab(
            actual_values,
            predicted_values,
            rownames=["Actual"],
            colnames=["Predicted"]
        )

        print(f"Confusion Matrix: {confusion_matrix}")
        print("\n")


    def _predict_row(self, row):

        state_scores = {}
        unseen = self._get_unseen_categories_in_row(row)
        print(unseen)

        for state in self.class_states:
            score = math.log(state.prob)
            for col in self.categorical_columns:
                k= self.k_per_feature[col]
                score += state.categorical_log_prob(col, row[col], k)

            for col in self.continuous_columns:
                score += state.continuous_log_prob(col, float(row[col]))

            state_scores[state.label] = score

        #TODO: i think we are using lists to say that there is like many class options, but all formulas limit it to 2 so we should as well!


        score1 = state_scores[self.class_states[0].label]
        score2 = state_scores[self.class_states[1].label]

        ratio = math.exp(score1 - score2)
                
        return {
            "confidence": ratio,
            "prediction": max(state_scores, key=state_scores.get),
            "had_unseen": len(unseen) > 0
        }


    def _get_unseen_categories_in_row(self, row):
        unseen = []

        for col in self.categorical_columns:
            if row[col] not in self.seen_categories[col]:
                unseen.append((col, row[col]))

        return unseen


    #METHODS JUST REQUIRED FOR ANSWERING THE QUESTIONS FOR THE ASSIGNMENT:
    
    def print_prior_probs(self):
        for state in self.class_states:
            print(f"Pr({self.class_column}={state.label}) = {state.prob}")
        print("\n")

    def print_continuous_estimates(self):
        for state in self.class_states:
            print(f"Class: {state.label}")
            for feature in state.continuous_stats.keys():
                print(f"  Feature: {feature}")
                print(f"    mean: {state.continuous_stats[feature]['mean']}")
                print(f"    std dev: {math.sqrt(state.continuous_stats[feature]['variance'])}")
            print("\n")


    #METHODS FOR QUESTION 2:

    def get_unseen_count(self, test_df):
        print(f"Number of instances with at least one unseen category value: {len(test_df["HAS_UNSEEN"] == True)}")
        print("\n")

    def get_high_confidence_instances(self, test_df, count, classification):
        classified = test_df[test_df["PREDICTED_INCOME"] == classification]
        if classification == "<=50K":
            classified = classified.sort_values(by="CONFIDENCE", ascending=False)
        elif classification == ">50K":
            classified = classified.sort_values(by="CONFIDENCE", ascending=True)

        return classified.head(count)

    def get_near_decision_boundary_instances(self, test_df, count):
        return test_df.iloc[(test_df["CONFIDENCE"] - 1).abs().argsort()].head(count)



class ClassState():
    def __init__(self, label: str, data_set, total_count):
        self.label = label

        self.total_count = total_count
        self.data_set = data_set

        self.count = len(data_set)
        self.prob = len(data_set) / total_count

        self.categorical_counts = {}
        self.continuous_stats = {}

    def categorical_log_prob(self, column: str, value, num_categories: int, alpha: float = LAPLACE) -> float:
        counts = self.categorical_counts[column]
        cnt = counts.get(value, 0)
        return math.log(cnt + alpha) - math.log(self.count + alpha * num_categories)

    def continuous_log_prob(self, column: str, x: float) -> float:
        mu = self.continuous_stats[column]["mean"]
        var = self.continuous_stats[column]["variance"]
        if var == 0:
            var = EPSILON
        return -0.5 * math.log(2 * math.pi * var) - ((x - mu) ** 2) / (2 * var)


CLASS_COLUMN = "income"
CLASS_VALUES = ["<=50K", ">50K"]
CATEGORICAL_COLUMNS = ["workclass","education","marital-status","occupation","relationship","race","sex",
"native-country"]
CONTINUOUS_COLUMNS = ["age","education-num","capital-gain","capital-loss","hours-per-week"]

# 1. read in data and clean empty rows
train_df = pd.read_csv("data/stripped-adult_supervised_train.csv")
#TODO: need to change drop method i think, and how do we do this throughout?
train_df = train_df.dropna()


#should really split into like load and train methods but whatever
model = NaiveBayesModel(CLASS_COLUMN, CLASS_VALUES, CATEGORICAL_COLUMNS, CONTINUOUS_COLUMNS, train_df)
model.train()


"""

QUESTION 1 OF THE ASSIGNMENT:

"""

#1. Prior probs:
model.print_prior_probs()

#2. for each continous feature, show mean and std dev for each class. Which continuous features show the greatest seperation between classes? 
model.print_continuous_estimates()


#3. for each categorical feature, identify the category value most strongle predicitve of each class. One way to measure this is the probability ratio R = P(xj = v | c1) / P(xj = v | c2). List the five most predictive category values for each class (across all categorical features) and their R values

#TODO: do this, seems a little tedious?


"""

QUESTION 2 OF THE ASSIGNMENT:

"""

test_df = pd.read_csv("data/stripped-adult_test.csv")
test_df = test_df.dropna()


#1. Report the overall accuracy as well as a class-level breakdown. Include confusion matrix and report precision, recall and f1-score for each class
#currently all this stuff is just in model.test().... TODO: this should probably change, some class should like store it!

model.test(test_df)

#2. Report how many test instances contain at least one unseen category value. How do we handle them. Were any test instances skipped entirely because they could not be classified?

model.get_unseen_count(test_df)


#3. Provide examples of instances classified with high and low confidnce. The condifence of a prediction can be measured using the ratio R = P(c1|x)/P(c2|x). Provide at least 3 instances of:
#a. instances classified as > 50k with high confidence

df_3a = model.get_high_confidence_instances(test_df, 3, ">50K")
print("Instances classified as >50K with high confidence:")
print("\n")
print(df_3a)
print("\n")


#b. instances classified as <=50k with high condifence

df_3b = model.get_high_confidence_instances(test_df, 3, "<=50K")
print("Instances classified as <=50K with high confidence:")
print("\n")
print(df_3b)
print("\n")

#c. instances near the decision boundary (R~1)

df_3b = model.get_near_decision_boundary_instances(test_df, 3)
print("Instances near decision boundary (R~1):")
print("\n")
print(df_3b)
print("\n")