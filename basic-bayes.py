import pandas as pd
import numpy as np

LAPLACE_SMOOTH = 1

class ContinuousFeature():
    def __init__(self, title: str, df: pd.DataFrame):
        self.title = title
        self.values = df[title].tolist()
        self.mu = np.mean(self.values)
        self.variance = np.var(self.values)


class CategoricalFeature():
    def __init__(self, title: str, df: pd.DataFrame):
        self.title = title
        self.values = df[title].tolist()
        self.probabilities = dict()
        
        self.calculate_probs()

    def calculate_probs(self):

        n = len(self.values)
        k = len(set(self.values))

        for value in set(self.values):
            self.probabilities[value] = (self.values.count(value) + LAPLACE_SMOOTH) / (n + LAPLACE_SMOOTH * k)
            


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
above_continuous_features = []
below_continuous_features = []

CATEGORICAL_FEATURE_TITLES = ["workclass","education","marital-status","occupation","relationship","race","sex",
"native-country"]
above_categorical_features = []
below_categorical_features = []

for feature_title in CONTINUOUS_FEATURE_TITLES:
    above_continuous_features.append(ContinuousFeature(feature_title, df_above))
    below_continuous_features.append(ContinuousFeature(feature_title, df_below))

for feature_title in CATEGORICAL_FEATURE_TITLES:
    above_categorical_features.append(CategoricalFeature(feature_title, df_above))
    below_categorical_features.append(CategoricalFeature(feature_title, df_below))


def predict_row()