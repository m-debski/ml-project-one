import pandas as pd
import numpy as np
import math

LAPLACE_SMOOTH = 1
EPSILON = 1e-9

class ContinuousFeature():
    def __init__(self, title: str, df: pd.DataFrame):
        self.title = title
        self.values = df[title].tolist()
        self.mu = np.mean(self.values)
        self.variance = np.var(self.values)
        if self.variance == 0:
            self.variance = EPSILON

    def log_probability(self, value: float):
        #gausian formula
        return -0.5 * math.log(2 * math.pi * self.variance) - ((value - self.mu) ** 2) / (2 * self.variance)


class CategoricalFeature():
    def __init__(self, title: str, df: pd.DataFrame):
        self.title = title
        self.values = df[title].tolist()
        self.probabilities = dict()
        self.n = 0
        self.k = 0
        
        self.calculate_probs()

    def calculate_probs(self):

        self.n = len(self.values)
        self.k = len(set(self.values))

        for value in set(self.values):
            self.probabilities[value] = (self.values.count(value) + LAPLACE_SMOOTH) / (self.n + LAPLACE_SMOOTH * self.k)


    def log_probability(self, value: str):
        # Handle unseen categories at prediction time using Laplace-smoothed fallback.
        probability = self.probabilities.get(value, LAPLACE_SMOOTH / (self.n + LAPLACE_SMOOTH * self.k))
        return math.log(probability)            


# 1. read in data and clean empty rows

df = pd.read_csv('data/stripped-adult_supervised_train.csv')
#TODO: change dropna method but yeah
df = df.dropna()

# 2. split the data into the two classes of income, so two dfs for each i guess

df_above = df[df["income"] == ">50K"]
print(len(df_above))
df_below = df[df["income"] == "<=50K"]
print(len(df_below))

# 3. Compute the class priors. Find proporiton of class to total, i.e P(>50k) and P(<50k)
p_above = len(df_above) / len(df)
print(f"P(above)={p_above}")
p_below = len(df_below) / len(df)
print(f"P(below)={p_below}")

# 4. For each class, then for each feature:
# Compute mean and variance for continuous features cause we need that normal distrubtion
# Use laplace smoothing and get probabilities for all the categorical ones

#TODO: this stuff could clearly easily be classed. like a DataSet class or smth containign the df and all these things

CONTINUOUS_FEATURE_TITLES = ["age","education-num","capital-gain","capital-loss","hours-per-week"]
above_continuous_features = {}
below_continuous_features = {}

CATEGORICAL_FEATURE_TITLES = ["workclass","education","marital-status","occupation","relationship","race","sex",
"native-country"]
above_categorical_features = {}
below_categorical_features = {}

for feature_title in CONTINUOUS_FEATURE_TITLES:
    above_continuous_features[feature_title] = ContinuousFeature(feature_title, df_above)
    below_continuous_features[feature_title] = ContinuousFeature(feature_title, df_below)

for feature_title in CATEGORICAL_FEATURE_TITLES:
    above_categorical_features[feature_title] = CategoricalFeature(feature_title, df_above)
    below_categorical_features[feature_title] = CategoricalFeature(feature_title, df_below)


def score_row(row):
    """Log unnormalized posterior terms: log P(c) + sum_j log P(x_j|c). Same additive constant for both."""
    above_score = math.log(p_above)
    for feature_title in CONTINUOUS_FEATURE_TITLES:
        above_score += above_continuous_features[feature_title].log_probability(row[feature_title])
    for feature_title in CATEGORICAL_FEATURE_TITLES:
        above_score += above_categorical_features[feature_title].log_probability(row[feature_title])

    below_score = math.log(p_below)
    for feature_title in CONTINUOUS_FEATURE_TITLES:
        below_score += below_continuous_features[feature_title].log_probability(row[feature_title])
    for feature_title in CATEGORICAL_FEATURE_TITLES:
        below_score += below_categorical_features[feature_title].log_probability(row[feature_title])

    return above_score, below_score


def posterior_ratio_R(row):
    """
    R = P(>50K | x) / P(<=50K | x).  The marginal P(x) cancels, so this equals exp(above_log - below_log).
    R >> 1: confident >50K; R << 1: confident <=50K; R ~ 1: near the decision boundary.
    """
    a, b = score_row(row)
    return math.exp(a - b)


def predict_row(row):
    above_score, below_score = score_row(row)
    if above_score > below_score:
        return ">50K"
    else:
        return "<=50K"



test = df.iloc[0]
prediction = predict_row(test)

print("actual:", test["income"])
print("learnt:", prediction)



new_data = pd.read_csv("data/stripped-adult_test.csv")
new_data = new_data.dropna()
new_data["PREDICTED_INCOME"] = new_data.apply(predict_row, axis=1)

print(new_data)


#calculate metrics
actual_values = new_data["income"]
predicted_values = new_data["PREDICTED_INCOME"]

ABOVE_true_positive = ((actual_values == ">50K") & (predicted_values == ">50K")).sum()
ABOVE_true_negative = ((actual_values == "<=50K") & (predicted_values == "<=50K")).sum()
ABOVE_false_positive = ((actual_values == "<=50K") & (predicted_values == ">50K")).sum()
ABOVE_false_negative = ((actual_values == ">50K") & (predicted_values == "<=50K")).sum()


BELOW_true_positive = ABOVE_true_negative
BELOW_true_negative = ABOVE_true_positive
BELOW_false_positive = ABOVE_false_negative
BELOW_false_negative = ABOVE_false_positive


true_positive = ABOVE_true_positive + BELOW_true_positive
true_negative = ABOVE_true_negative + BELOW_true_negative
false_positive = ABOVE_false_positive + BELOW_false_positive
false_negative = ABOVE_false_negative + BELOW_false_negative

accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)


precision_above = ABOVE_true_positive / (ABOVE_true_positive + ABOVE_false_positive)
precision_below = BELOW_true_positive/ (BELOW_true_positive + BELOW_false_positive)

recall_above = ABOVE_true_positive / (ABOVE_true_positive + ABOVE_false_negative)
recall_below = BELOW_true_positive / (BELOW_true_positive + BELOW_false_negative)

f1_above = 2*(precision_above*recall_above) / (precision_above+recall_above)
f1_below = 2*(precision_below*recall_below) / (precision_below+recall_below)

print(f"Accuracy: {accuracy}")
print(f"Precision ABOVE: {precision_above}")
print(f"Precision BELOW: {precision_below}")
print(f"Recall ABOVE: {recall_above}")
print(f"Recall BELOW: {recall_below}")
print(f"F1 ABOVE: {f1_above}")
print(f"F1 BELOW: {f1_below}")

confusion_matrix = pd.DataFrame(
    [[ABOVE_true_positive, ABOVE_false_negative],
     [ABOVE_false_positive, ABOVE_true_negative]],
    index=["Actual >50K", "Actual <=50K"],
    columns=["Pred >50K", "Pred <=50K"]
)

print(confusion_matrix)

#some categorical feature values in the test set may not have appeared during training, how many test instances contain at least one
#unseen category value. Where any test instances skipped because they could not be classified?

#TODO: need to implement a way to find this



#provide examples of instances classified with high and low confidence. Measured using R = P(c1|
#x) / P(c2|x)
#Higher R means the model is more confident the instance belongs to C1. R ~ 1 means its 
#uncertain. provide various examples the thingy lists

#TODO: need to implement this


"""

now we need to extend the model for part 3 using option 1 label propgation

- use q1 model to assign predicted labels ot the unlabeeled dataset. these are a ground truth
- retrain the model on the combined supervises data and pseudo-labeled data.

"""