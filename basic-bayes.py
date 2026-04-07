import pandas as pd
import numpy as np



class ContinuousFeature():
    def __init__(self, title: str, df: pd.DataFrame):
        self.title = title
        self.values = df[title].tolist()
        self.mu = np.mean(self.values)
        self.sigma = np.std(self.values)
class CategoricalFeature():

# 1. read in data and clean empty rows

df = pd.read_csv('data/stripped-adult_supervised_train.csv')
#TODO: change dropna method but yeah
df_cleaned = df.dropna(axis=1, how='all')

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

CONTINUOUS_FEATURE_TITLES = ["age","education-num","capital-gain","capital-loss","hours-per-week"]
continuous_features = []
CATEGORICAL_FEATURE_TITLES = ["workclass","education","education-num","marital-status","occupation","relationship","race","sex",
"native-country"]
categorical_features = []

for feature_title in CONTINUOUS_FEATURE_TITLES:
    continuous_features.append(ContinuousFeature(feature_title, df_above))

for feature_title in CATEGORICAL_FEATURE_TITLES:
    categorical_features.append(CategoricalFeature(feature_title))


