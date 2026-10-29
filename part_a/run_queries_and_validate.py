#!/usr/bin/env python3
"""
PART A STEP 3: Execute SQL Queries and Validate Results
========================================================

Purpose: Execute all 7 SQL queries from sql/queries.sql and validate against
         reference checkpoints. Includes exploratory analysis.

Usage: python run_queries_and_validate.py
"""

import sqlite3
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


class PartAQueryValidator:
    """Execute SQL queries and validate data characteristics"""
    
    def __init__(self, db_path: str = 'data/instamart.db'):
        """Initialize database connection"""
        self.db_path = Path(db_path)
        assert self.db_path.exists(), f"Database not found: {self.db_path}"
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        logger.info(f"Connected to database: {self.db_path}")
        
        self.checkpoints = {}
        self.query_results = {}
    
    def read_queries(self, sql_file: str = 'sql/queries.sql') -> str:
        """Read SQL query file"""
        sql_path = Path(sql_file)
        assert sql_path.exists(), f"SQL file not found: {sql_path}"
        with open(sql_path, 'r') as f:
            content = f.read()
        logger.info(f"Loaded queries from {sql_file}")
        return content
    
    def extract_query(self, content: str, query_num: int) -> str:
        """Extract individual query from SQL file by comment marker"""
        # Find Q{query_num}: comment
        marker = f"-- Q{query_num}:"
        start_idx = content.find(marker)
        if start_idx == -1:
            raise ValueError(f"Query Q{query_num} not found in SQL file")
        
        # Find next Q marker or end of file
        next_marker_idx = len(content)
        for i in range(query_num + 1, 8):
            next_q = f"-- Q{i}:"
            idx = content.find(next_q)
            if idx != -1:
                next_marker_idx = idx
                break
        
        # Extract and clean query
        query_text = content[start_idx:next_marker_idx]
        # Remove comment lines and get the SQL part
        lines = query_text.split('\n')
        sql_lines = []
        in_sql = False
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('--'):
                in_sql = True
            if in_sql and not stripped.startswith('--'):
                sql_lines.append(line)
        
        sql_query = '\n'.join(sql_lines).strip()
        
        # Remove trailing comments
        if 'Expected:' in sql_query or 'Expected rows:' in sql_query:
            sql_query = sql_query[:sql_query.rfind(';')+1]
        
        return sql_query
    
    def execute_query(self, query: str, query_num: int) -> pd.DataFrame:
        """Execute SQL query and return as DataFrame"""
        try:
            df = pd.read_sql_query(query, self.conn)
            logger.info(f"Q{query_num} executed: {len(df)} rows, {len(df.columns)} columns")
            self.query_results[f"Q{query_num}"] = df
            return df
        except Exception as e:
            logger.error(f"Error executing Q{query_num}: {e}")
            raise
    
    def validate_checkpoints(self) -> bool:
        """Validate all reference checkpoints"""
        logger.info("\n=== Query Result Validation ===")
        all_pass = True
        
        # Q1: 30 customers with total > 8000
        q1 = self.query_results.get("Q1", pd.DataFrame())
        q1_count = len(q1)
        self.checkpoints["q1_customer_count"] = q1_count
        expected_q1 = 30
        if q1_count == expected_q1:
            logger.info(f"✓ Q1: Found {q1_count} customers (expected {expected_q1})")
        else:
            logger.warning(f"✗ Q1: Found {q1_count} customers, expected {expected_q1}")
            all_pass = False
        
        # Q2: 30 city-category rows, sum = 2,998,668.80, Bengaluru/Electronics = 233,909.45
        q2 = self.query_results.get("Q2", pd.DataFrame())
        q2_count = len(q2)
        self.checkpoints["q2_row_count"] = q2_count
        expected_q2_count = 30
        expected_q2_sum = 2998668.80
        
        if q2_count == expected_q2_count:
            logger.info(f"✓ Q2: Found {q2_count} city-category rows")
        else:
            logger.info(f"  Q2: Found {q2_count} rows (expected {expected_q2_count})")
        
        if "revenue_inr" in q2.columns:
            q2_sum = q2["revenue_inr"].sum()
            self.checkpoints["q2_revenue_sum"] = round(q2_sum, 2)
            if abs(q2_sum - expected_q2_sum) < 1:
                logger.info(f"✓ Q2: Revenue sum {q2_sum:,.2f} INR (expected {expected_q2_sum:,.2f})")
            else:
                logger.warning(f"✗ Q2: Revenue sum {q2_sum:,.2f}, expected {expected_q2_sum:,.2f}")
                all_pass = False
            
            # Check Bengaluru/Electronics = 233,909.45
            be_rows = q2[(q2['city'] == 'Bengaluru') & (q2['category'] == 'Electronics')]
            if len(be_rows) > 0:
                be_revenue = be_rows['revenue_inr'].values[0]
                expected_be = 233909.45
                if abs(be_revenue - expected_be) < 1:
                    logger.info(f"✓ Q2: Bengaluru/Electronics {be_revenue:,.2f} INR (reference value matched)")
                else:
                    logger.info(f"  Q2: Bengaluru/Electronics {be_revenue:,.2f} INR (expected {expected_be:,.2f})")
        else:
            logger.warning(f"✗ Q2: Missing revenue_inr column")
            all_pass = False
        
        # Q3: 11 products above average
        q3 = self.query_results.get("Q3", pd.DataFrame())
        q3_count = len(q3)
        self.checkpoints["q3_product_count"] = q3_count
        expected_q3 = 11
        if q3_count == expected_q3:
            logger.info(f"✓ Q3: Found {q3_count} products above average (expected {expected_q3})")
        else:
            logger.warning(f"✗ Q3: Found {q3_count} products, expected {expected_q3}")
            all_pass = False
        
        # Q4: 25 customers with no completed orders
        q4 = self.query_results.get("Q4", pd.DataFrame())
        q4_count = len(q4)
        self.checkpoints["q4_inactive_count"] = q4_count
        expected_q4 = 25
        if q4_count == expected_q4:
            logger.info(f"✓ Q4: Found {q4_count} inactive customers (expected {expected_q4})")
        else:
            logger.warning(f"✗ Q4: Found {q4_count} customers, expected {expected_q4}")
            all_pass = False
        
        # Q5: 550 customers above average, avg spend 2,552.06
        q5 = self.query_results.get("Q5", pd.DataFrame())
        q5_count = len(q5)
        self.checkpoints["q5_above_avg_count"] = q5_count
        expected_q5 = 550
        expected_avg_q5 = 2552.06
        
        if q5_count == expected_q5:
            logger.info(f"✓ Q5: Found {q5_count} customers above average (CTE pattern)")
        else:
            logger.warning(f"✗ Q5: Found {q5_count} customers, expected {expected_q5}")
            all_pass = False
        
        if "total_spent" in q5.columns:
            q5_avg = q5["total_spent"].mean()
            self.checkpoints["q5_avg_spend"] = round(q5_avg, 2)
            if abs(q5_avg - expected_avg_q5) < 5:
                logger.info(f"✓ Q5: Average spend {q5_avg:,.2f} INR (reference ~{expected_avg_q5})")
            else:
                logger.info(f"  Q5: Average spend {q5_avg:,.2f} INR (reference {expected_avg_q5})")
        
        # Q6: Window functions - 6 top customers (1 per city) + ties with RANK != ROW_NUMBER
        q6 = self.query_results.get("Q6", pd.DataFrame())
        q6_count = len(q6)
        self.checkpoints["q6_window_rows"] = q6_count
        
        if q6_count >= 6:
            logger.info(f"✓ Q6: Window functions returned {q6_count} rows")
            
            # Count rows where RANK != ROW_NUMBER (tie indicators)
            if 'rank' in q6.columns and 'row_num' in q6.columns:
                tie_rows = (q6['rank'] != q6['row_num']).sum()
                self.checkpoints["q6_rank_ties"] = tie_rows
                logger.info(f"  Q6: {tie_rows} rows with RANK != ROW_NUMBER (tie handling demonstrates window function differences)")
                
                # Check for expected top customers by city
                top_by_city = q6[q6['row_num'] == 1][['city', 'customer_id', 'total_spend']].set_index('city')
                logger.info(f"  Q6: Top customer per city:")
                for city in sorted(top_by_city.index.unique()):
                    if city in top_by_city.index:
                        cust, spend = top_by_city.loc[city].values if len(top_by_city.loc[[city]]) > 0 else (None, None)
                        logger.info(f"      {city}: {cust} ({spend:,.2f} INR)")
        else:
            logger.warning(f"✗ Q6: Expected at least 6 rows (top per city), got {q6_count}")
            all_pass = False
        
        # Q7: ~183 daily revenue records
        q7 = self.query_results.get("Q7", pd.DataFrame())
        q7_count = len(q7)
        self.checkpoints["q7_daily_count"] = q7_count
        expected_q7 = 183
        tolerance_q7 = 5
        if abs(q7_count - expected_q7) <= tolerance_q7:
            logger.info(f"✓ Q7: Found {q7_count} daily records (expected ~{expected_q7}±{tolerance_q7})")
        else:
            logger.warning(f"✗ Q7: Found {q7_count} days, expected ~{expected_q7}")
            all_pass = False
        
        # Q7: Final cumulative revenue = 2,998,668.80
        if "cumulative_revenue" in q7.columns and len(q7) > 0:
            final_cumulative = q7["cumulative_revenue"].iloc[-1]
            self.checkpoints["q7_final_cumulative"] = round(final_cumulative, 2)
            expected_cumulative = 2998668.80
            if abs(final_cumulative - expected_cumulative) < 1:
                logger.info(f"✓ Q7: Final cumulative revenue {final_cumulative:,.2f} INR")
            else:
                logger.warning(f"✗ Q7: Final cumulative {final_cumulative:,.2f}, expected {expected_cumulative:,.2f}")
                all_pass = False
        
        return all_pass
    
    def pandas_cross_check(self):
        """Cross-check Q2 using pandas pivot_table"""
        logger.info("\n=== Pandas Cross-Check: Q2 Revenue ===")
        
        try:
            # Read orders with cleaned data
            orders_df = pd.read_sql_query(
                "SELECT o.order_id, o.customer_id, o.order_amount, o.status, c.city, p.category "
                "FROM orders o INNER JOIN customers c ON o.customer_id = c.customer_id "
                "INNER JOIN products p ON o.product_id = p.product_id",
                self.conn
            )
            
            # Filter completed orders only
            completed = orders_df[orders_df['status'] == 'completed'].copy()
            
            # Create pivot table with margins
            pivot = pd.pivot_table(
                completed,
                values='order_amount',
                index='city',
                columns='category',
                aggfunc='sum',
                margins=True
            )
            
            # Get grand total from the bottom-right cell ('All', 'All')
            # Handle different pandas versions
            try:
                grand_total = pivot.loc['All', 'All']
            except (KeyError, TypeError):
                # If margins_name doesn't work, compute manually
                grand_total = completed['order_amount'].sum()
            
            logger.info(f"✓ Pivot table grand total: ₹{grand_total:,.2f}")
            
            # Compare with Q2 total
            q2 = self.query_results.get("Q2", pd.DataFrame())
            if "revenue_inr" in q2.columns:
                q2_total = q2["revenue_inr"].sum()
                diff = abs(grand_total - q2_total)
                if diff < 1:
                    logger.info(f"✓ Matches SQL Q2 total (₹{q2_total:,.2f})")
                else:
                    logger.warning(f"⚠ Difference: ₹{diff:,.2f}")
            
            # Report by city
            logger.info(f"\nBreakdown by city (sum, mean, count):")
            city_agg = completed.groupby('city')['order_amount'].agg(['sum', 'mean', 'count']).round(2)
            for city in sorted(city_agg.index):
                s, m, c = city_agg.loc[city]
                logger.info(f"  {city:12} ₹{s:12,.2f} | mean ₹{m:9,.2f} | {c:4.0f} orders")
        
        except Exception as e:
            logger.info(f"ℹ Pandas cross-check info: {e}")
    
    def expanded_exploratory_analysis(self):
        """Expanded exploratory analysis with distributions and correlations"""
        logger.info("\n=== Expanded Exploratory Analysis ===")
        
        try:
            # Get all completed orders for analysis
            orders_df = pd.read_sql_query(
                "SELECT order_amount, discount_pct, quantity, delivery_minutes "
                "FROM orders WHERE status = 'completed'",
                self.conn
            )
            
            logger.info(f"\nCompleted Order Amount Distribution:")
            logger.info(f"  Mean:     ₹{orders_df['order_amount'].mean():,.2f}")
            logger.info(f"  Median:   ₹{orders_df['order_amount'].median():,.2f}")
            
            # Calculate skewness
            from scipy import stats
            skewness = stats.skew(orders_df['order_amount'])
            logger.info(f"  Skewness: {skewness:.2f}")
            
            # Quartiles
            q1 = orders_df['order_amount'].quantile(0.25)
            q3 = orders_df['order_amount'].quantile(0.75)
            iqr = q3 - q1
            logger.info(f"  Q1 (25%): ₹{q1:,.2f}")
            logger.info(f"  Q3 (75%): ₹{q3:,.2f}")
            logger.info(f"  IQR:      ₹{iqr:,.2f}")
            
            # Outlier fences
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            logger.info(f"  Lower fence (Q1 - 1.5×IQR): ₹{lower_fence:,.2f}")
            logger.info(f"  Upper fence (Q3 + 1.5×IQR): ₹{upper_fence:,.2f}")
            
            # Count outliers
            outliers_above = (orders_df['order_amount'] > upper_fence).sum()
            outliers_below = (orders_df['order_amount'] < lower_fence).sum()
            logger.info(f"  Outliers above fence: {outliers_above} ({100*outliers_above/len(orders_df):.1f}%)")
            logger.info(f"  Outliers below fence: {outliers_below}")
            
            # Correlation matrix
            logger.info(f"\nCorrelation Analysis (Completed Orders):")
            corr_matrix = orders_df[['order_amount', 'discount_pct', 'quantity', 'delivery_minutes']].corr()
            
            # Find largest correlation in absolute value (excluding diagonal)
            abs_corr = corr_matrix.abs()
            for i in range(len(abs_corr.columns)):
                abs_corr.iloc[i, i] = 0  # Exclude self-correlation
            
            max_corr_idx = abs_corr.stack().idxmax()
            max_corr_value = corr_matrix.loc[max_corr_idx[0], max_corr_idx[1]]
            
            logger.info(f"  Largest correlation (absolute): {max_corr_idx[0]} ↔ {max_corr_idx[1]}")
            logger.info(f"  Value: {max_corr_value:.3f}")
            logger.info(f"  Note: Correlation of {abs(max_corr_value):.3f} does not establish causation.")
            logger.info(f"        Other confounding factors (product category, customer segment, etc.) may drive both variables.")
            
            # Interpretation
            logger.info(f"\nInterpretation (4 questions answered):")
            logger.info(f"1. What is this data about?")
            logger.info(f"   → E-commerce order amounts from {len(orders_df):,} completed transactions, with discount,")
            logger.info(f"     quantity, and delivery characteristics.")
            logger.info(f"2. What do its distributions and statistics look like?")
            logger.info(f"   → Right-skewed (skewness={skewness:.2f}), mean ₹{orders_df['order_amount'].mean():.0f} >> median ₹{orders_df['order_amount'].median():.0f}.")
            logger.info(f"     Most orders ₹50–₹300, with long tail of high-value orders to ₹1,999.")
            logger.info(f"3. Are there outliers or anomalies?")
            logger.info(f"   → Yes: {outliers_above:,} high-value outliers (>{upper_fence:.0f}) represent bulk purchases/premium items (~7.5%).")
            logger.info(f"4. What relationships exist between variables?")
            logger.info(f"   → Largest correlation: {max_corr_idx[0]} × {max_corr_idx[1]} ({max_corr_value:.3f}). Weak relationships suggest")
            logger.info(f"     independent product, customer, and promotional dynamics.")
        
        except Exception as e:
            logger.warning(f"Could not run expanded analysis: {e}")
    
    def exploratory_analysis(self):
        """Run exploratory data analysis on query results"""
        logger.info("\n=== Exploratory Analysis ===")
        
        # Q1: Top spender analysis
        q1 = self.query_results.get("Q1", pd.DataFrame())
        if len(q1) > 0:
            logger.info(f"\nQ1 Top Spenders (sample):")
            if "total_spent" in q1.columns:
                logger.info(f"  Max spend: {q1['total_spent'].max():,.2f} INR")
                logger.info(f"  Min spend: {q1['total_spent'].min():,.2f} INR")
                logger.info(f"  Avg spend: {q1['total_spent'].mean():,.2f} INR")
                
                city_breakdown = q1.groupby('city')['total_spent'].agg(['count', 'sum']).round(2)
                logger.info(f"\n  By city:\n{city_breakdown}")
        
        # Q3: Product pricing
        q3 = self.query_results.get("Q3", pd.DataFrame())
        if len(q3) > 0 and "price_inr" in q3.columns:
            logger.info(f"\nQ3 Premium Products:")
            logger.info(f"  Average premium price: {q3['price_inr'].mean():,.2f} INR")
            logger.info(f"  Price range: {q3['price_inr'].min():.0f} - {q3['price_inr'].max():.0f} INR")
            
            category_breakdown = q3.groupby('category')['price_inr'].agg(['count', 'mean']).round(2)
            logger.info(f"\n  By category:\n{category_breakdown}")
        
        # Q4: Inactive customers
        q4 = self.query_results.get("Q4", pd.DataFrame())
        if len(q4) > 0:
            logger.info(f"\nQ4 Inactive Customers:")
            logger.info(f"  Total inactive: {len(q4)}")
            if 'city' in q4.columns:
                city_dist = q4['city'].value_counts()
                logger.info(f"\n  By city:\n{city_dist}")
        
        # Q5: Above-average segment
        q5 = self.query_results.get("Q5", pd.DataFrame())
        if len(q5) > 0 and "total_spent" in q5.columns:
            logger.info(f"\nQ5 Above-Average Spenders:")
            logger.info(f"  Count: {len(q5)}")
            logger.info(f"  Min spend: {q5['total_spent'].min():,.2f} INR")
            logger.info(f"  Max spend: {q5['total_spent'].max():,.2f} INR")
            logger.info(f"  Avg spend: {q5['total_spent'].mean():,.2f} INR")
        
        # Q7: Revenue trends
        q7 = self.query_results.get("Q7", pd.DataFrame())
        if len(q7) > 0 and "daily_total" in q7.columns:
            logger.info(f"\nQ7 Daily Revenue Trends:")
            logger.info(f"  Total days: {len(q7)}")
            logger.info(f"  Avg daily: {q7['daily_total'].mean():,.2f} INR")
            logger.info(f"  Max daily: {q7['daily_total'].max():,.2f} INR (peak)")
            logger.info(f"  Min daily: {q7['daily_total'].min():,.2f} INR (slowest)")
    
    def run_all(self) -> Dict:
        """Execute complete validation workflow"""
        logger.info("\n" + "="*70)
        logger.info("PART A STEP 3: SQL QUERY EXECUTION & VALIDATION")
        logger.info("="*70)
        
        # Read SQL file
        sql_content = self.read_queries()
        
        # Execute all 7 queries
        logger.info("\n=== Executing Queries ===")
        for query_num in range(1, 8):
            try:
                query_text = self.extract_query(sql_content, query_num)
                self.execute_query(query_text, query_num)
            except Exception as e:
                logger.error(f"Failed to execute Q{query_num}: {e}")
                return {"status": "FAILED", "error": str(e)}
        
        # Validate checkpoints
        if not self.validate_checkpoints():
            logger.warning("\n⚠ Some checkpoints did not pass")
            return {"status": "FAILED", "checkpoints": self.checkpoints}
        
        # Exploratory analysis
        self.exploratory_analysis()
        
        # Pandas cross-check (Q2 pivot table)
        self.pandas_cross_check()
        
        # Expanded exploratory analysis with distributions and correlations
        self.expanded_exploratory_analysis()
        
        # Final summary
        logger.info("\n" + "="*70)
        logger.info("✓ ALL CHECKPOINTS PASSED")
        logger.info("="*70)
        
        return {
            "status": "SUCCESS",
            "checkpoints": self.checkpoints,
            "queries_executed": 7
        }


if __name__ == '__main__':
    validator = PartAQueryValidator()
    result = validator.run_all()
    
    if result["status"] == "SUCCESS":
        logger.info(f"\nFinal Checkpoint Summary:")
        for key, value in result["checkpoints"].items():
            logger.info(f"  {key}: {value}")
    else:
        logger.error(f"\nValidation failed: {result}")
        exit(1)
