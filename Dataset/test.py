import pandas as pd

df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nLabel Counts:")
print(df[" Label"].value_counts())

print("\nFirst 5 Rows:")
print(df.head())