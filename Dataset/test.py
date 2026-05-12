import pandas as pd

df = pd.read_csv("Combined_Dataset/CICIDS2017_GROUPED.csv")

features = df.drop("Label", axis=1).columns

print("Total features:", len(features))
print(features.tolist())