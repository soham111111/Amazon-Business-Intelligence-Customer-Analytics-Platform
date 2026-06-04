import pandas as pd

df = pd.read_csv("data/Amazon_transaction_data.csv")

print("Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())