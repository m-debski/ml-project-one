import pandas as pd
import numpy as np
import math

LAPLACE = 1

class NaiveBayesModel():
    def __init__(self, class_column, class_values, categorical_columns, continuous_columns, data_set):
        self.data_set = data_set
        self.class_column = class_column #hard coded title of column that is for classes
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





class ClassState():
    def __init__(self, label: str, data_set, total_count):
        self.label = label

        self.total_count = total_count
        self.data_set = data_set

        self.count = len(data_set)
        self.prob = len(data_set) / total_count

        self.categorical_counts = {}
        self.continuous_stats = {}

        


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

test_df = pd.read_csv("data/stripped-adult_supervised_train.csv")
test_df = test_df.dropna("data/stripped-adult_test.csv")



#1. Report the overall accuracy as well as a class-level breakdown. Include confusion matrix and report precision, recall and f1-score for each class