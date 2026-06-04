SELECT *
FROM amazon_transactions_raw
LIMIT 10;

SELECT
COUNT(*) AS total_rows,
COUNT(OrderID) AS order_ids,
COUNT(CustomerID) AS customer_ids,
COUNT(ProductID) AS product_ids
FROM amazon_transactions_raw;