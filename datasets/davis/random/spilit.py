import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Load your dataset
df = pd.read_csv('full.csv')

# Verify total samples match expected
total_samples = 6011 + 2086 + 3006  # Should equal len(df)
assert len(df) == total_samples, f"Dataset has {len(df)} samples, but required {total_samples}"

# First split: separate out test set (3006 samples)
train_val_df, test_df = train_test_split(
    df,
    test_size=3006,
    random_state=SEED,
    stratify=df['Y'] if 'Y' in df.columns else None  # Stratify by target if available
)

# Second split: separate remaining into train (6011) and val (2086)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=2086,
    random_state=SEED,
    stratify=train_val_df['Y'] if 'Y' in df.columns else None
)

# Verify sizes
print(f"Training set size: {len(train_df)} (target ratio: {train_df['Y'].mean():.2%})")
print(f"Validation set size: {len(val_df)} (target ratio: {val_df['Y'].mean():.2%})")
print(f"Test set size: {len(test_df)} (target ratio: {test_df['Y'].mean():.2%})")

# Save to separate files
train_df.to_csv('train_set.csv', index=False)
val_df.to_csv('val_set.csv', index=False)
test_df.to_csv('test_set.csv', index=False)

print("\nSplitting completed successfully with the following distribution:")
print(f"Train: {len(train_df)} samples")
print(f"Val: {len(val_df)} samples")
print(f"Test: {len(test_df)} samples")