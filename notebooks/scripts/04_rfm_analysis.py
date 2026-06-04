import pandas as pd

df = pd.read_csv("data/Amazon_transaction_data.csv")

df['OrderDate'] = pd.to_datetime(df['OrderDate'])

snapshot_date = pd.Timestamp('2024-12-30')

rfm = df.groupby('CustomerID').agg({
    'OrderDate': lambda x: (snapshot_date - x.max()).days,
    'OrderID': 'count',
    'TotalAmount': 'sum'
})

rfm.columns = ['Recency', 'Frequency', 'Monetary']

rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])

rfm['RFM_Score'] = (
    rfm['R_Score'].astype(str) +
    rfm['F_Score'].astype(str) +
    rfm['M_Score'].astype(str)
)

print(rfm.head())

def segment_customer(row):
    if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
        return 'Champions'

    elif row['F_Score'] >= 4 and row['M_Score'] >= 3:
        return 'Loyal Customers'

    elif row['R_Score'] >= 4 and row['F_Score'] >= 3:
        return 'Potential Loyalists'

    elif row['R_Score'] <= 2 and row['F_Score'] >= 3:
        return 'At Risk'

    elif row['R_Score'] <= 2 and row['F_Score'] <= 2:
        return 'Lost Customers'

    else:
        return 'Others'


rfm['Segment'] = rfm.apply(segment_customer, axis=1)

segment_counts = rfm['Segment'].value_counts()

print("\nCustomer Segments:")
print(segment_counts)



#(import matplotlib.pyplot as plt

#segment_counts = rfm['Segment'].value_counts()

#plt.figure(figsize=(8,5))
#segment_counts.plot(kind='bar')

#plt.title("Customer Segmentation")
#plt.xlabel("Segment")
#plt.ylabel("Number of Customers")

#plt.tight_layout()
#plt.show()
rfm['Segment'].value_counts()

rfm.to_csv(
    r"C:\Users\SOHAM PANDAV\OneDrive\Desktop\Amazon_AI_BI_Platform\reports\rfm_segments.csv",
    index=True
)

print("RFM file saved successfully!")
