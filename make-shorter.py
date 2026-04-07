import pandas as pd

filename = input("enter file name in data dir: ")

df = pd.read_csv("data/" + filename, nrows=100)
df.to_csv(f"data/stripped-{filename}", index=False)