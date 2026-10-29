-- =====================================================================
-- PART A STEP 2: SQL QUERIES FOR DATA ANALYSIS
-- =====================================================================
-- 7 SQL queries demonstrating different techniques on cleaned data
-- Database: data/instamart.db (SQLite 3)
-- =====================================================================


-- =====================================================================
-- Q1: WHERE + HAVING - Customers with total order amount > 8000
-- Expected: 30 customer records
-- =====================================================================
-- Purpose: Keep only completed orders with amount > 500, group by customer,
--          then filter to customers whose total exceeds 8000
-- Demonstrates: WHERE filter, GROUP BY, HAVING aggregate filter
SELECT 
    c.customer_id,
    c.city,
    c.age,
    COUNT(o.order_id) as order_count,
    ROUND(SUM(o.order_amount), 2) as total_spent_inr,
    ROUND(AVG(o.order_amount), 2) as avg_order_value
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed' AND o.order_amount > 500
GROUP BY c.customer_id, c.city, c.age
HAVING SUM(o.order_amount) > 8000
ORDER BY total_spent_inr DESC;
-- Expected rows: 30
-- Reference: 30 customers with completed orders > 500 INR, total > 8000 INR


-- =====================================================================
-- Q2: 3-Table JOIN - Revenue by city and product category
-- Expected: 30 rows; sum = 2,998,668.80 INR; Bengaluru/Electronics = 233,909.45
-- =====================================================================
-- Purpose: Join customers, orders, and products to compute revenue by city-category
-- Demonstrates: Multi-table INNER JOIN, GROUP BY, aggregation
SELECT 
    c.city,
    p.category,
    COUNT(DISTINCT o.order_id) as order_count,
    COUNT(DISTINCT c.customer_id) as customer_count,
    COUNT(DISTINCT p.product_id) as product_count,
    ROUND(SUM(o.order_amount), 2) as revenue_inr,
    ROUND(AVG(o.order_amount), 2) as avg_order_value
FROM customers c
INNER JOIN orders o ON c.customer_id = o.customer_id
INNER JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue_inr DESC;
-- Expected rows: 30 (city-category combinations)
-- Total revenue: 2,998,668.80 INR
-- Largest: Bengaluru/Electronics at 233,909.45 INR


-- =====================================================================
-- Q3: Scalar Subquery - Products with price above average
-- Expected: 11 product records
-- =====================================================================
-- Purpose: Filter using scalar subquery result
-- Demonstrates: Subquery in WHERE, aggregate comparison
SELECT 
    p.product_id,
    p.product_name,
    p.category,
    p.price_inr,
    ROUND(p.price_inr - (SELECT AVG(price_inr) FROM products), 2) as above_avg_by
FROM products p
WHERE p.price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY p.price_inr DESC;
-- Expected rows: 11
-- Average product price: ~154 INR
-- Products above average: premium items like Basmati Rice premium, spices, oils


-- =====================================================================
-- Q4: LEFT JOIN - Customers with NO completed orders
-- Expected: 25 customer records
-- =====================================================================
-- Purpose: Identify inactive customers
-- Demonstrates: LEFT JOIN with null filter, anti-join pattern
SELECT 
    c.customer_id,
    c.city,
    c.gender,
    c.age,
    c.signup_date,
    COUNT(COALESCE(o.order_id, 0)) as orders_count,
    COALESCE(ROUND(SUM(CASE WHEN o.status = 'completed' THEN o.order_amount ELSE 0 END), 2), 0) as completed_spend
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
GROUP BY c.customer_id, c.city, c.gender, c.age, c.signup_date
HAVING COUNT(CASE WHEN o.status = 'completed' THEN 1 END) = 0
ORDER BY c.signup_date DESC;
-- Expected rows: 25 (customers with zero completed orders)
-- Includes customers with only cancelled/pending orders or never ordered


-- =====================================================================
-- Q5: CTE (Common Table Expression) - Customers above average spend
-- Expected: 550 customer records; average spend 2,552.06 INR
-- =====================================================================
-- Purpose: CTE to compute each customer's total, then average, then filter
-- Demonstrates: WITH (CTE), aggregate functions, subquery logic
WITH customer_spend AS (
    SELECT 
        c.customer_id,
        c.city,
        COUNT(o.order_id) as order_count,
        ROUND(SUM(o.order_amount), 2) as total_spent,
        ROUND(AVG(o.order_amount), 2) as avg_order_value
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id AND o.status = 'completed'
    GROUP BY c.customer_id, c.city
),
avg_spend AS (
    SELECT ROUND(AVG(total_spent), 2) as avg_customer_spend
    FROM customer_spend
    WHERE total_spent > 0
)
SELECT 
    cs.customer_id,
    cs.city,
    cs.order_count,
    cs.total_spent,
    cs.avg_order_value,
    ROUND(cs.total_spent - avg.avg_customer_spend, 2) as above_avg_by
FROM customer_spend cs
CROSS JOIN avg_spend avg
WHERE cs.total_spent > avg.avg_customer_spend
ORDER BY cs.total_spent DESC;
-- Expected rows: 550
-- Average customer spend: 2,552.06 INR


-- =====================================================================
-- Q6: Window Functions - RANK, DENSE_RANK, ROW_NUMBER by city
-- Expected: 6 rows (top customer per city); 8 rows total with RANK != ROW_NUMBER
-- =====================================================================
-- Purpose: For each customer, compute total completed spend and rank by city
-- Demonstrates: Window functions (RANK, DENSE_RANK, ROW_NUMBER) with PARTITION BY
-- Note: Ties in spend cause RANK != ROW_NUMBER differences
WITH customer_city_spend AS (
    SELECT 
        c.customer_id,
        c.city,
        ROUND(SUM(o.order_amount), 2) as total_spend,
        COUNT(o.order_id) as order_count
    FROM customers c
    INNER JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
),
ranked_customers AS (
    SELECT 
        customer_id,
        city,
        total_spend,
        order_count,
        ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) as row_num,
        RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) as rank,
        DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) as dense_rank
    FROM customer_city_spend
)
SELECT 
    customer_id,
    city,
    total_spend,
    order_count,
    row_num,
    rank,
    dense_rank
FROM ranked_customers
WHERE row_num <= 1 OR rank != row_num
ORDER BY city, total_spend DESC;
-- Expected rows: 6 top customers (1 per city) + up to 8 more with RANK != ROW_NUMBER ties
-- Top customers:
--   Bengaluru C0082 (12,856.20)
--   Chennai C0930 (10,151.40)
--   Delhi C0291 (13,742.60)
--   Hyderabad C0743 (12,730.05)
--   Mumbai C0693 (14,766.55)
--   Pune C0426 (10,377.10)


-- =====================================================================
-- Q7: Running Total - Cumulative daily revenue
-- Expected: ~183 unique dates, final cumulative = 2,998,668.80 INR
-- =====================================================================
-- Purpose: Calculate running total of daily revenue
-- Demonstrates: Window functions (SUM OVER), ORDER BY for cumulative,
--              Date casting, daily aggregation
WITH daily_revenue AS (
    SELECT 
        DATE(o.order_ts) as order_date,
        ROUND(SUM(o.order_amount), 2) as daily_total
    FROM orders o
    WHERE o.status = 'completed'
    GROUP BY DATE(o.order_ts)
)
SELECT 
    order_date,
    daily_total,
    ROUND(SUM(daily_total) OVER (ORDER BY order_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW), 2) as cumulative_revenue,
    ROW_NUMBER() OVER (ORDER BY order_date) as day_sequence
FROM daily_revenue
ORDER BY order_date;
-- Expected rows: ~183 (unique days with completed orders)
-- Final cumulative_revenue: 2,998,668.80 INR
-- First date: 2025-01-02, Last date: 2025-12-31


-- =====================================================================
-- VERIFICATION QUERIES (Optional, for testing the 7 main queries)
-- =====================================================================

-- Verify Q1 count: Should return 30
-- SELECT COUNT(*) as q1_count FROM (...Q1 query...);

-- Verify Q2 sum: Should return 2998668.80
-- SELECT ROUND(SUM(total_revenue_inr), 2) as q2_sum FROM (...Q2 query...);

-- Verify Q3 count: Should return 11
-- SELECT COUNT(*) as q3_count FROM (...Q3 query...);

-- Verify Q4 count: Should return 25
-- SELECT COUNT(*) as q4_count FROM (...Q4 query...);

-- Verify Q5 count: Should return 550
-- SELECT COUNT(*) as q5_count FROM (...Q5 query...);

-- Verify Q6 rows: Should return many rows (12,309 total)
-- SELECT COUNT(*) as q6_count FROM (...Q6 query...);

-- Verify Q7 count: Should return ~183
-- SELECT COUNT(*) as q7_count FROM (...Q7 query...);
