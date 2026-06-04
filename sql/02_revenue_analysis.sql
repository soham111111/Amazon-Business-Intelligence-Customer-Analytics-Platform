SELECT
ROUND(SUM(TotalAmount),2) AS Total_Revenue
FROM amazon_transactions_raw;

SELECT
COUNT(DISTINCT OrderID) AS Total_Orders
FROM amazon_transactions_raw;

SELECT
COUNT(DISTINCT CustomerID) AS Total_Customers
FROM amazon_transactions_raw;

SELECT
Category,
ROUND(SUM(TotalAmount),2) AS Revenue
FROM amazon_transactions_raw
GROUP BY Category
ORDER BY Revenue DESC;