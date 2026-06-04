import pandas as pd

df = pd.read_csv("data/amazon.csv")

print("\nTop Rated Products:")
print(
    df.groupby("product_name")["rating"]
      .mean()
      .sort_values(ascending=False)
      .head(10)
)

print("\nMost Reviewed Products:")
print(
    df.groupby("product_name")["rating_count"]
      .max()
      .sort_values(ascending=False)
      .head(10)
)