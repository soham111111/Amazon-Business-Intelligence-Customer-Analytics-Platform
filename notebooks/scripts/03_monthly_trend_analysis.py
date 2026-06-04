import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/Amazon_transaction_data.csv")

df['OrderDate'] = pd.to_datetime(df['OrderDate'])

monthly_sales = (
    df.groupby(df['OrderDate'].dt.to_period('M'))
      ['TotalAmount']
      .sum()
)

print(monthly_sales)

monthly_sales.plot(figsize=(10,5))
plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()