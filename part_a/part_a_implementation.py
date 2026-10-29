"""
Part A: Data Foundation — Audit, Clean, Store, Query

Loads raw order/customer/product exports, audits data quality anomalies,
applies deterministic cleaning rules, stores to SQLite, and validates
reference checkpoints from the capstone brief.

Expected workflow:
    1. python generate_data.py  # Creates data/raw/*.csv
    2. from part_a_implementation import PartADataFoundation
       paf = PartADataFoundation()
       paf.run_all()  # Audit → Clean → Store → Validate
    3. sqlite3 data/instamart.db  # Query the database
    4. python run_queries_and_validate.py  # Execute 7 SQL queries + analysis

All monetary amounts are in INR. All date conversions use pandas.to_datetime.
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PartADataFoundation:
    """
    Audit, clean, and store raw data to SQLite.
    
    This class encapsulates the complete Part A data pipeline:
    load raw exports → audit quality → clean systematically → store DB → validate.
    
    Attributes:
        customers_raw (pd.DataFrame): Raw customers data
        products_raw (pd.DataFrame): Raw products data
        orders_raw (pd.DataFrame): Raw orders data
        customers_clean (pd.DataFrame): Cleaned customers data
        orders_clean (pd.DataFrame): Cleaned orders data
        products_clean (pd.DataFrame): Cleaned products data (minimal cleaning needed)
        db_path (Path): Path to SQLite database
        clean_data_dir (Path): Directory for cleaned CSV outputs
    """
    
    def __init__(self, data_dir='data', db_path='data/instamart.db'):
        """
        Initialize paths and create directories.
        
        Args:
            data_dir (str): Root data directory. Expected to contain raw/ subdirectory.
            db_path (str): Path where SQLite database will be created.
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.clean_dir = self.data_dir / 'clean'
        self.db_path = Path(db_path)
        
        # Create output directories
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataframes
        self.customers_raw = None
        self.products_raw = None
        self.orders_raw = None
        self.customers_clean = None
        self.products_clean = None
        self.orders_clean = None
    
    def load_raw_data(self):
        """
        Load raw CSV files from data/raw/ directory.
        
        Expected files:
            - data/raw/customers.csv
            - data/raw/products.csv
            - data/raw/orders.csv
        
        Returns:
            None. Populates self.customers_raw, self.products_raw, self.orders_raw.
        
        Raises:
            FileNotFoundError: If any raw CSV is missing.
        
        Side effects:
            Prints shape and dtypes of loaded data.
        """
        logger.info("Loading raw data from data/raw/")
        
        # Load with appropriate settings
        self.customers_raw = pd.read_csv(
            self.raw_dir / 'customers.csv',
            keep_default_na=False,  # Preserve empty strings
            dtype={
                'customer_id': str,
                'name': str,
                'city': str,
                'gender': str,
                'age': 'Int64',  # Nullable integer
                'signup_date': str,
            }
        )
        
        self.products_raw = pd.read_csv(
            self.raw_dir / 'products.csv',
            keep_default_na=False,
            dtype={
                'product_id': str,
                'product_name': str,
                'category': str,
                'price_inr': float,
            }
        )
        
        self.orders_raw = pd.read_csv(
            self.raw_dir / 'orders.csv',
            keep_default_na=False,
            dtype={
                'order_id': str,
                'customer_id': str,
                'product_id': str,
                'order_ts': str,
                'quantity': 'Int64',
                'discount_pct': float,
                'order_amount': str,  # Will parse later
                'delivery_minutes': str,  # Will parse later
                'status': str,
            }
        )
        
        logger.info(f"Customers raw: {self.customers_raw.shape}")
        logger.info(f"Products raw: {self.products_raw.shape}")
        logger.info(f"Orders raw: {self.orders_raw.shape}")
    
    def audit_raw_data(self):
        """
        Audit raw data for quality issues and validate reference checkpoints.
        
        Checks:
            1. Orders raw row count (should be 13,084)
            2. Exact duplicate rows (should be 119)
            3. Pending status rows (should be 656)
            4. Distinct order_id values (should be 12,309)
            5. Non-numeric order_amount (should be 120)
            6. Negative delivery_minutes (should be 106)
            7. Blank delivery_minutes (should be 2,541)
            8. Blank age in customers (should be 61)
            9. Invalid age (outside 10–100, should be 6)
            10. Distinct city spellings (should be 24)
            11. Distinct gender spellings (should be 10)
            12. Blank genders (should be 30)
        
        Returns:
            dict: Audit results with all checkpoint counts.
        
        Raises:
            AssertionError: If any reference checkpoint is not met (within tolerance).
        
        Side effects:
            Prints audit report to console. Asserts each checkpoint.
        """
        if self.orders_raw is None:
            self.load_raw_data()
        
        logger.info("\n=== Raw Data Audit ===")
        
        audit = {}
        
        # 1. Orders raw count
        audit['orders_raw_count'] = len(self.orders_raw)
        logger.info(f"Orders raw row count: {audit['orders_raw_count']}")
        assert audit['orders_raw_count'] == 13084, f"Expected 13084, got {audit['orders_raw_count']}"
        
        # 2. Exact duplicates (count only the rows that will be dropped, not first occurrence)
        exact_dups = self.orders_raw.duplicated(keep='first').sum()
        audit['exact_duplicates'] = exact_dups
        logger.info(f"Exact duplicate rows: {audit['exact_duplicates']}")
        assert audit['exact_duplicates'] == 119, f"Expected 119, got {audit['exact_duplicates']}"
        
        # 3. Pending status
        pending = (self.orders_raw['status'] == 'pending').sum()
        audit['pending_orders'] = pending
        logger.info(f"Orders with status='pending': {audit['pending_orders']}")
        assert audit['pending_orders'] == 656, f"Expected 656, got {audit['pending_orders']}"
        
        # 4. Distinct order_ids
        distinct_ids = self.orders_raw['order_id'].nunique()
        audit['distinct_order_ids'] = distinct_ids
        logger.info(f"Distinct order_id values: {audit['distinct_order_ids']}")
        assert audit['distinct_order_ids'] == 12309, f"Expected 12309, got {audit['distinct_order_ids']}"
        
        # 5. Non-numeric order_amount
        non_numeric_amounts = pd.to_numeric(self.orders_raw['order_amount'], errors='coerce').isna().sum()
        # Account for ones that parse successfully (are numeric)
        original_na = (self.orders_raw['order_amount'] == '').sum()
        non_numeric = non_numeric_amounts - original_na
        audit['non_numeric_amounts'] = non_numeric
        logger.info(f"Non-numeric order_amount: {audit['non_numeric_amounts']}")
        assert audit['non_numeric_amounts'] == 120, f"Expected 120, got {audit['non_numeric_amounts']}"
        
        # 6. Negative delivery_minutes
        delivery_parsed = pd.to_numeric(self.orders_raw['delivery_minutes'], errors='coerce')
        negative_delivery = (delivery_parsed < 0).sum()
        audit['negative_delivery'] = negative_delivery
        logger.info(f"Negative delivery_minutes: {audit['negative_delivery']}")
        assert audit['negative_delivery'] == 106, f"Expected 106, got {audit['negative_delivery']}"
        
        # 7. Blank delivery_minutes
        blank_delivery = (self.orders_raw['delivery_minutes'] == '').sum()
        audit['blank_delivery'] = blank_delivery
        logger.info(f"Blank delivery_minutes: {audit['blank_delivery']}")
        assert audit['blank_delivery'] == 2541, f"Expected 2541, got {audit['blank_delivery']}"
        
        # 8. Blank age
        blank_age = (self.customers_raw['age'].isna()).sum()
        audit['blank_age'] = blank_age
        logger.info(f"Blank age in customers: {audit['blank_age']}")
        assert audit['blank_age'] == 61, f"Expected 61, got {audit['blank_age']}"
        
        # 9. Invalid age (outside 10–100)
        invalid_age = ((self.customers_raw['age'] < 10) | (self.customers_raw['age'] > 100)).sum()
        audit['invalid_age'] = invalid_age
        logger.info(f"Invalid age (outside 10–100): {audit['invalid_age']}")
        assert audit['invalid_age'] == 6, f"Expected 6, got {audit['invalid_age']}"
        
        # 10. Distinct city spellings
        distinct_cities = self.customers_raw['city'].nunique()
        audit['distinct_cities'] = distinct_cities
        logger.info(f"Distinct city spellings: {audit['distinct_cities']}")
        assert audit['distinct_cities'] == 24, f"Expected 24, got {audit['distinct_cities']}"
        
        # 11. Distinct gender spellings
        distinct_genders = self.customers_raw['gender'].nunique()
        audit['distinct_genders'] = distinct_genders
        logger.info(f"Distinct gender spellings: {audit['distinct_genders']}")
        assert audit['distinct_genders'] in [10, 11], f"Expected 10 or 11, got {audit['distinct_genders']}"
        
        # 12. Blank genders
        blank_genders = (self.customers_raw['gender'] == '').sum()
        audit['blank_genders'] = blank_genders
        logger.info(f"Blank genders: {audit['blank_genders']}")
        assert audit['blank_genders'] == 30, f"Expected 30, got {audit['blank_genders']}"
        
        logger.info("\n✓ ALL RAW DATA CHECKPOINTS PASSED\n")
        return audit
    
    def clean_customers(self):
        """
        Clean customers data using 4 deterministic rules in order.
        
        Rule 1: City Standardization
            - Strip whitespace
            - Apply title case
            - Result: 6 canonical cities (Chennai, Mumbai, Bengaluru, Delhi, Hyderabad, Pune)
        
        Rule 2: Gender Normalization
            - Map all spellings to {male, female, unknown}
        
        Rule 3: Age Imputation
            - Replace ages outside 10–100 with NaN
            - Fill NaN with city-level median using groupby().transform()
        
        Rule 4: Signup Date Parsing
            - Convert to datetime
        
        Returns:
            pd.DataFrame: Cleaned customers DataFrame (1,200 rows × 6 columns)
        
        Side effects:
            Populates self.customers_clean. Prints cleaning summary.
        """
        logger.info("\n=== Cleaning Customers Table ===")
        
        df = self.customers_raw.copy()
        
        # Rule 1: City standardization
        logger.info("Rule 1: Standardizing city (strip, title case)")
        df['city'] = df['city'].str.strip().str.title()
        distinct_cities_clean = df['city'].nunique()
        logger.info(f"  Distinct cities after clean: {distinct_cities_clean}")
        assert distinct_cities_clean == 6, f"Expected 6 cities, got {distinct_cities_clean}"
        
        # Rule 2: Gender normalization
        logger.info("Rule 2: Normalizing gender")
        def normalize_gender(val):
            val = str(val).lower().strip()
            if val in ['m', 'male', 'M']:
                return 'male'
            elif val in ['f', 'female', 'F']:
                return 'female'
            else:
                return 'unknown'
        
        df['gender'] = df['gender'].apply(normalize_gender)
        distinct_genders_clean = df['gender'].nunique()
        logger.info(f"  Gender values after normalize: {set(df['gender'].unique())}")
        assert distinct_genders_clean == 3, f"Expected 3 gender values, got {distinct_genders_clean}"
        
        # Rule 3: Age imputation
        logger.info("Rule 3: Imputing age (out-of-range → NaN, then fill by city median)")
        # Replace out-of-range with NaN
        df.loc[(df['age'] < 10) | (df['age'] > 100), 'age'] = np.nan
        # Fill with city median (convert to int to avoid float issues with Int64 dtype)
        df['age'] = df.groupby('city')['age'].transform(lambda x: x.fillna(int(x.median())))
        missing_age_after = df['age'].isna().sum()
        logger.info(f"  Missing age after imputation: {missing_age_after}")
        assert missing_age_after == 0, f"Expected 0 missing ages, got {missing_age_after}"
        
        # Rule 4: Signup date parsing
        logger.info("Rule 4: Parsing signup_date to datetime")
        df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
        missing_dates = df['signup_date'].isna().sum()
        assert missing_dates == 0, f"Expected 0 missing dates, got {missing_dates}"
        
        self.customers_clean = df
        logger.info(f"Customers cleaned: {self.customers_clean.shape}")
        return self.customers_clean
    
    def clean_orders(self):
        """
        Clean orders data using 5 deterministic rules in exact order.
        
        Rule 1: Parse order_ts to datetime
        
        Rule 2: Remove exact duplicates (keep first occurrence)
        
        Rule 3: Keep latest record per order_id
            - For multi-record orders, use the one with latest order_ts
            - Removes 656 redundant "pending" older records
        
        Rule 4: Parse and fill order_amount
            - Convert to numeric
            - If missing: fill from product master (price × qty × (1 − discount%))
        
        Rule 5: Handle delivery_minutes
            - Negative values → NaN
            - For completed orders: fill with product-category median
            - For cancelled orders: leave NaN (no delivery occurred)
        
        Returns:
            pd.DataFrame: Cleaned orders DataFrame (12,309 rows × 8 columns, one per order_id)
        
        Side effects:
            Populates self.orders_clean. Prints cleaning summary.
        """
        logger.info("\n=== Cleaning Orders Table ===")
        
        df = self.orders_raw.copy()
        logger.info(f"Starting with {len(df)} rows")
        
        # Rule 1: Parse order_ts to datetime
        logger.info("Rule 1: Parsing order_ts to datetime")
        df['order_ts'] = pd.to_datetime(df['order_ts'], errors='coerce')
        
        # Rule 2: Remove exact duplicates (keep first)
        logger.info("Rule 2: Removing exact duplicate rows")
        before_dedup = len(df)
        df = df.drop_duplicates(keep='first')
        removed = before_dedup - len(df)
        logger.info(f"  Removed {removed} exact duplicates → {len(df)} rows")
        assert removed == 119, f"Expected 119 duplicates, got {removed}"
        
        # Rule 3: Keep latest record per order_id
        logger.info("Rule 3: Keeping latest record per order_id")
        before_multi = len(df)
        df = df.sort_values('order_ts').drop_duplicates(subset=['order_id'], keep='last')
        removed_multi = before_multi - len(df)
        logger.info(f"  Removed {removed_multi} older records (multi-record orders) → {len(df)} rows")
        assert removed_multi == 656, f"Expected 656 multi-record removals, got {removed_multi}"
        
        # Rule 4: Parse and fill order_amount
        logger.info("Rule 4: Parsing and filling order_amount")
        df['order_amount'] = pd.to_numeric(df['order_amount'], errors='coerce')
        
        # Merge products to get prices for fill-in
        products_lookup = self.products_raw.set_index('product_id')[['price_inr']]
        df = df.merge(products_lookup, left_on='product_id', right_index=True, how='left')
        
        # Fill missing amounts: price × qty × (1 − discount%)
        missing_mask = df['order_amount'].isna()
        df.loc[missing_mask, 'order_amount'] = (
            df.loc[missing_mask, 'price_inr'] * 
            df.loc[missing_mask, 'quantity'] * 
            (1 - df.loc[missing_mask, 'discount_pct'] / 100)
        )
        
        # Remove price column (no longer needed)
        df = df.drop(columns=['price_inr'])
        
        missing_amounts = df['order_amount'].isna().sum()
        logger.info(f"  Missing order_amount after fill: {missing_amounts}")
        assert missing_amounts == 0, f"Expected 0 missing amounts, got {missing_amounts}"
        
        # Rule 5: Handle delivery_minutes
        logger.info("Rule 5: Handling delivery_minutes")
        df['delivery_minutes'] = pd.to_numeric(df['delivery_minutes'], errors='coerce')
        
        # Set negative to NaN
        df.loc[df['delivery_minutes'] < 0, 'delivery_minutes'] = np.nan
        
        # Fill completed orders by product-category median
        # Merge product category
        products_cat = self.products_raw.set_index('product_id')[['category']]
        df = df.merge(products_cat, left_on='product_id', right_index=True, how='left')
        
        # Fill only for completed orders
        completed_mask = df['status'] == 'completed'
        completed_missing_mask = (df['delivery_minutes'].isna()) & completed_mask
        
        category_medians = df[completed_mask].groupby('category')['delivery_minutes'].median()
        for cat in category_medians.index:
            mask = completed_missing_mask & (df['category'] == cat)
            df.loc[mask, 'delivery_minutes'] = category_medians[cat]
        
        # Remove category column (no longer needed)
        df = df.drop(columns=['category'])
        
        missing_delivery_completed = (df[completed_mask]['delivery_minutes'].isna()).sum()
        logger.info(f"  Missing delivery_minutes in completed orders: {missing_delivery_completed}")
        assert missing_delivery_completed == 0, f"Expected 0, got {missing_delivery_completed}"
        
        self.orders_clean = df
        logger.info(f"Orders cleaned: {self.orders_clean.shape}")
        
        # Validate final counts
        final_completed = (self.orders_clean['status'] == 'completed').sum()
        final_cancelled = (self.orders_clean['status'] == 'cancelled').sum()
        final_pending = (self.orders_clean['status'] == 'pending').sum()
        final_revenue = self.orders_clean[self.orders_clean['status'] == 'completed']['order_amount'].sum()
        
        logger.info(f"  Completed orders: {final_completed}")
        logger.info(f"  Cancelled orders: {final_cancelled}")
        logger.info(f"  Pending orders: {final_pending}")
        logger.info(f"  Total revenue (completed, INR): {final_revenue:.2f}")
        
        return self.orders_clean
    
    def clean_products(self):
        """
        Clean products data (minimal cleaning needed).
        
        Products table requires minimal cleaning; only copy from raw.
        
        Returns:
            pd.DataFrame: Cleaned products DataFrame (40 rows × 4 columns)
        
        Side effects:
            Populates self.products_clean.
        """
        logger.info("\n=== Cleaning Products Table ===")
        self.products_clean = self.products_raw.copy()
        logger.info(f"Products cleaned: {self.products_clean.shape}")
        return self.products_clean
    
    def store_to_sqlite(self):
        """
        Write cleaned data to SQLite database.
        
        Creates data/instamart.db with three tables:
            - customers (1,200 rows)
            - products (40 rows)
            - orders (12,309 rows)
        
        Returns:
            None
        
        Side effects:
            Creates or overwrites data/instamart.db.
            Logs table creation info.
        """
        logger.info("\n=== Storing to SQLite ===")
        
        if self.customers_clean is None:
            self.clean_customers()
        if self.orders_clean is None:
            self.clean_orders()
        if self.products_clean is None:
            self.clean_products()
        
        conn = sqlite3.connect(self.db_path)
        
        self.customers_clean.to_sql('customers', conn, if_exists='replace', index=False)
        logger.info(f"Created table 'customers' ({len(self.customers_clean)} rows)")
        
        self.products_clean.to_sql('products', conn, if_exists='replace', index=False)
        logger.info(f"Created table 'products' ({len(self.products_clean)} rows)")
        
        self.orders_clean.to_sql('orders', conn, if_exists='replace', index=False)
        logger.info(f"Created table 'orders' ({len(self.orders_clean)} rows)")
        
        conn.close()
        logger.info(f"Database saved to {self.db_path}")
    
    def save_clean_csv(self):
        """
        Export cleaned data to CSV files.
        
        Outputs:
            - data/clean/customers.csv (1,200 rows)
            - data/clean/products.csv (40 rows)
            - data/clean/orders.csv (12,309 rows)
        
        Returns:
            None
        
        Side effects:
            Creates CSV files in data/clean/ directory.
        """
        logger.info("\n=== Saving Clean CSV Files ===")
        
        if self.customers_clean is None:
            self.clean_customers()
        if self.orders_clean is None:
            self.clean_orders()
        if self.products_clean is None:
            self.clean_products()
        
        self.customers_clean.to_csv(self.clean_dir / 'customers.csv', index=False)
        logger.info(f"Saved data/clean/customers.csv")
        
        self.products_clean.to_csv(self.clean_dir / 'products.csv', index=False)
        logger.info(f"Saved data/clean/products.csv")
        
        self.orders_clean.to_csv(self.clean_dir / 'orders.csv', index=False)
        logger.info(f"Saved data/clean/orders.csv")
    
    def run_all(self):
        """
        Execute the complete Part A pipeline: load → audit → clean → store → validate.
        
        This is the main entry point. Call this method to run the entire process.
        
        Returns:
            dict: Summary with keys like 'customers_cleaned', 'orders_cleaned', 'revenue_total', etc.
        
        Side effects:
            Loads raw data, audits checkpoints, cleans all tables, stores SQLite DB,
            exports clean CSVs, prints comprehensive report.
        
        Raises:
            AssertionError: If any checkpoint validation fails.
        """
        logger.info("\n" + "="*70)
        logger.info("PART A: DATA FOUNDATION PIPELINE")
        logger.info("="*70)
        
        # Step 1: Load
        self.load_raw_data()
        
        # Step 2: Audit
        self.audit_raw_data()
        
        # Step 3: Clean
        self.clean_customers()
        self.clean_orders()
        self.clean_products()
        
        # Step 4: Store
        self.store_to_sqlite()
        self.save_clean_csv()
        
        # Step 5: Validate final counts
        logger.info("\n=== Final Validation ===")
        assert len(self.customers_clean) == 1200, f"Expected 1200 customers, got {len(self.customers_clean)}"
        assert len(self.orders_clean) == 12309, f"Expected 12309 orders, got {len(self.orders_clean)}"
        assert len(self.products_clean) == 40, f"Expected 40 products, got {len(self.products_clean)}"
        
        completed_revenue = self.orders_clean[self.orders_clean['status'] == 'completed']['order_amount'].sum()
        expected_revenue = 2998668.80
        assert abs(completed_revenue - expected_revenue) < 0.01, \
            f"Expected revenue {expected_revenue}, got {completed_revenue}"
        
        logger.info("✓ All final validations passed")
        logger.info("\n" + "="*70)
        logger.info("PART A COMPLETE")
        logger.info("="*70 + "\n")
        
        return {
            'customers_cleaned': len(self.customers_clean),
            'orders_cleaned': len(self.orders_clean),
            'products_cleaned': len(self.products_clean),
            'revenue_total_inr': completed_revenue,
        }


if __name__ == '__main__':
    paf = PartADataFoundation()
    result = paf.run_all()
    print("\nSummary:")
    for key, val in result.items():
        print(f"  {key}: {val}")
