import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split

# =========================
# LOAD DATA
# =========================
folder = "../Dataset/"

files = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

dfs = []

for f in files:
    path = os.path.join(folder, f)

    print(f"Loading: {f}")

    df = pd.read_csv(path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Ensure label exists
    if "Label" not in df.columns:
        print(f"Skipping {f}")
        continue

    dfs.append(df)

# =========================
# ALIGN COLUMNS
# =========================
common_columns = set(dfs[0].columns)

for df in dfs[1:]:
    common_columns = common_columns.intersection(set(df.columns))

common_columns = list(common_columns)

dfs = [df[common_columns] for df in dfs]

# =========================
# MERGE DATASETS
# =========================
full_df = pd.concat(dfs, ignore_index=True)

print("\nRAW SHAPE:", full_df.shape)

# =========================
# CLEAN DATA
# =========================
full_df.replace([np.inf, -np.inf], np.nan, inplace=True)

full_df.dropna(inplace=True)

full_df.drop_duplicates(inplace=True)

print("\nAFTER CLEANING SHAPE:", full_df.shape)

# =========================
# CLEAN LABELS
# =========================
full_df["Label"] = full_df["Label"].astype(str)

full_df["Label"] = full_df["Label"].str.strip()

# =========================
# GROUP LABELS
# =========================
def map_labels(label):

    label = str(label).strip()

    if label == "BENIGN":
        return "BENIGN"

    elif label in [
        "DoS Hulk",
        "DoS GoldenEye",
        "DoS slowloris",
        "DoS Slowhttptest",
        "DDoS"
    ]:
        return "DOS_ATTACK"

    elif label == "PortScan":
        return "RECON_ATTACK"

    elif label in [
        "Bot",
        "Infiltration"
    ]:
        return "ADVANCED_ATTACK"

    elif label in [
        "Web Attack - Brute Force",
        "Web Attack - XSS",
        "Web Attack - Sql Injection",
        "Web Attack � Brute Force",
        "Web Attack � XSS",
        "Web Attack � Sql Injection"
    ]:
        return "WEB_ATTACK"

    elif label in [
        "FTP-Patator",
        "SSH-Patator"
    ]:
        return "BRUTE_FORCE"

    else:
        return "OTHER"


full_df["Label"] = full_df["Label"].apply(map_labels)

print("\nGROUPED LABELS:")
print(full_df["Label"].value_counts())

# =========================
# REMOVE OTHER
# =========================
full_df = full_df[full_df["Label"] != "OTHER"]

print("\nAFTER REMOVING OTHER:")
print(full_df["Label"].value_counts())

# =========================
# BALANCE DATASET
# =========================
print("\nBALANCING DATASET...")

# Find smallest class count
min_count = full_df["Label"].value_counts().min()

balanced_dfs = []

for label in full_df["Label"].unique():

    class_df = full_df[full_df["Label"] == label]

    # Downsample larger classes
    if len(class_df) > min_count:
        class_df = class_df.sample(
            n=min_count,
            random_state=42
        )

    # Upsample smaller classes
    elif len(class_df) < min_count:
        class_df = class_df.sample(
            n=min_count,
            replace=True,
            random_state=42
        )

    balanced_dfs.append(class_df)

# Merge balanced data
full_df = pd.concat(balanced_dfs)

# Shuffle dataset
full_df = full_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nBALANCED LABELS:")
print(full_df["Label"].value_counts())

# =========================
# TRAIN / TEST SPLIT
# =========================
train_df, test_df = train_test_split(
    full_df,
    test_size=0.2,
    random_state=42,
    stratify=full_df["Label"]
)

print("\nTRAIN SHAPE:", train_df.shape)
print("TEST SHAPE:", test_df.shape)

# =========================
# CREATE OUTPUT FOLDER
# =========================
output_folder = "../Dataset/Combined_Dataset"

os.makedirs(output_folder, exist_ok=True)

# =========================
# SAVE FILES
# =========================
full_output = os.path.join(
    output_folder,
    "CICIDS2017_GROUPED.csv"
)

train_output = os.path.join(
    output_folder,
    "CICIDS2017_TRAIN.csv"
)

test_output = os.path.join(
    output_folder,
    "CICIDS2017_TEST.csv"
)

full_df.to_csv(full_output, index=False)

train_df.to_csv(train_output, index=False)

test_df.to_csv(test_output, index=False)


# =========================
# FINAL OUTPUT
# =========================
print("\nFILES SAVED SUCCESSFULLY")

print("\nFULL DATASET:")
print(full_output)

print("\nTRAIN DATASET:")
print(train_output)

print("\nTEST DATASET:")
print(test_output)