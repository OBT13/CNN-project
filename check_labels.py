import pandas as pd

labels = pd.read_csv("data/csvTrainLabel 13440x1.csv", header=None)

print("Min label:", labels.min().values)
print("Max label:", labels.max().values)
print("Unique labels:", labels.nunique().values)
