import pandas as pd

# Read only the first 100 rows (this automatically includes the header)
df = pd.read_csv('data/qa_dataset_composition.csv', nrows=100)

# Save to a new CSV file
df.to_csv('data/first_100_rows_composition.csv', index=False)

print("Extraction complete!")