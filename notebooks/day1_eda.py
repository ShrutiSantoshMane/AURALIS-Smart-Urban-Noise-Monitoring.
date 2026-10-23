import pandas as pd

metadata_path = "../data/UrbanSound8K.csv"

df = pd.read_csv(metadata_path)

print(df.head())
print("Number of samples:", len(df))
print("Number of classes:", df["class"].nunique())
print(df["class"].unique())
import matplotlib.pyplot as plt

class_counts = df["class"].value_counts()

print("\nClass Distribution:")
print(class_counts)

plt.figure(figsize=(10, 5))

class_counts.plot(kind="bar")

plt.title("UrbanSound8K Class Distribution")
plt.xlabel("Sound Class")
plt.ylabel("Number of Audio Clips")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("../results/class_distribution.png", dpi=300)

plt.show()
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:")
print(df.duplicated().sum())
import os

audio_folder = "../data"

print("\nChecking audio folders:")

for fold in range(1, 11):
    folder = os.path.join(audio_folder, f"fold{fold}")

    if os.path.exists(folder):
        files = os.listdir(folder)
        print(f"fold{fold}: {len(files)} audio files")
    else:
        print(f"fold{fold}: NOT FOUND")
        import pandas as pd
        import numpy as np
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt
        import os