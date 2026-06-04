import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/Amazon_transaction_data.csv")

category_sales = (
    df.groupby("Category")["TotalAmount"]
      .sum()
      .sort_values(ascending=False)
)

print(category_sales)

category_sales.plot(kind="bar")
plt.title("Revenue by Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()