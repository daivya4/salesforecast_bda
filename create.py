import pandas as pd
import os

# Paths (based on your project structure)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

input_path = os.path.join(DATA_DIR, "train.csv")
output_path = os.path.join(DATA_DIR, "store_1_2_3_2017.csv")

print(f"Reading from: {input_path}")

# Load full dataset
df = pd.read_csv(input_path)

# Convert date column
df['date'] = pd.to_datetime(df['date'])

# Filter: Stores 1, 2, 3 + 2017 only
df_filtered = df[
    (df['store_nbr'].isin([1, 2, 3])) &
    (df['date'].dt.year == 2017)
]

print("Unique stores in filtered data:", df_filtered['store_nbr'].unique())

# Save new file
df_filtered.to_csv(output_path, index=False)

print(f"Saved filtered dataset to: {output_path}")
print(f"Total rows: {len(df_filtered)}")