import pandas as pd
import numpy as np
import os

np.random.seed(42)

num_samples = 2000

data = []

for _ in range(num_samples):
    age = np.random.randint(55, 90)
    gender = np.random.randint(0, 2)  # 0 = Female, 1 = Male
    mmse = np.random.randint(10, 30)  # Cognitive score

    # Simple logic to simulate Alzheimer severity
    if mmse > 26:
        label = 0  # NonDemented
    elif 22 < mmse <= 26:
        label = 1  # VeryMild
    elif 16 < mmse <= 22:
        label = 2  # Mild
    else:
        label = 3  # Moderate

    data.append([age, gender, mmse, label])

df = pd.DataFrame(data, columns=["age", "gender", "mmse", "label"])

os.makedirs("data", exist_ok=True)
df.to_csv("data/clinical.csv", index=False)

print("Synthetic clinical dataset created successfully!")