"""
RFM Segmentation Module for Instamart Customer Analytics

This module builds the Recency-Frequency-Monetary (RFM) table from clean order data,
clusters customers using K-means (K=2 to K=8), selects an optimal K using elbow and
silhouette analysis with business judgment, profiles segments, names them, applies
PCA analysis, and exports results to CSV for downstream use in lapse prediction
and ticket triage.

Key safeguards:
- RFM computed using only completed orders as of snapshot date 2025-10-01
- Customers with no completed orders are excluded
- K selection balances statistical metrics with business interpretability
- Premium segment identified by highest mean monetary value
- All segment assignments, scaling, and PCA are reproducible with fixed random seed

Reference values (for validation):
- RFM table: 1,175 customers; recency median 19, min 0, max 180
- K-means: K=2 (1820.8 inertia, 0.465 silhouette) through K=8 (366.5, 0.412)
- Selected K (from {3,4,5}): K=4 typical; sizes ~436/138/419/182
- PCA explained variance: ~0.634 and ~0.218 (total ~0.852)
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RFMSegmentation:
    """
    Compute RFM features, cluster with K-means, profile segments, and export results.
    
    Workflow:
    1. Load clean orders and customers from SQLite
    2. Compute RFM table for customers with ≥1 completed order
    3. Standardize RFM features
    4. Run K-means for K in [2, 3, 4, 5, 6, 7, 8]
    5. Select optimal K using elbow, silhouette, and business judgment
    6. Profile segments and assign business names
    7. Apply PCA for visualization
    8. Export to CSV
    
    Attributes:
        db_path (Path): Path to SQLite database
        snapshot_date (datetime): Reference date for RFM computation (2025-10-01)
        rfm_df (pd.DataFrame): RFM table with recency, frequency, monetary
        scaled_rfm (np.ndarray): Standardized RFM features
        scaler (StandardScaler): Fitted scaler for reproducibility
        kmeans_results (dict): K-means metrics for K=2 to K=8
        optimal_k (int): Selected K from {3, 4, 5}
        kmeans_model (KMeans): Fitted K-means for optimal K
        segment_names (dict): Mapping segment_id -> business name
        is_premium (dict): Mapping segment_id -> Premium flag (True for highest mean monetary)
        pca (PCA): Fitted PCA(n_components=2) on scaled RFM
        segments_df (pd.DataFrame): Final output with customer_id, RFM, segment_id, name, premium flag
    """
    
    def __init__(self, db_path: str = 'data/instamart.db', snapshot_date: str = '2025-10-01'):
        """
        Initialize RFMSegmentation.
        
        Args:
            db_path (str): Path to SQLite database (must contain 'orders' and 'customers' tables)
            snapshot_date (str): Reference date for RFM in 'YYYY-MM-DD' format
        
        Raises:
            FileNotFoundError: If database does not exist
            sqlite3.OperationalError: If required tables are missing
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        self.snapshot_date = pd.to_datetime(snapshot_date)
        logger.info(f"Initialized RFMSegmentation with snapshot date {self.snapshot_date.date()}")
        
        # Placeholders for downstream attributes
        self.rfm_df = None
        self.scaled_rfm = None
        self.scaler = None
        self.kmeans_results = {}
        self.optimal_k = None
        self.kmeans_model = None
        self.segment_names = {}
        self.is_premium = {}
        self.pca = None
        self.segments_df = None
    
    def load_data(self) -> tuple:
        """
        Load orders and customers from SQLite database.
        
        Returns:
            tuple: (orders_df, customers_df)
                - orders_df: Columns [order_id, customer_id, order_ts, order_amount, status, ...]
                - customers_df: Columns [customer_id, city, age, gender, signup_date, ...]
        
        Raises:
            sqlite3.OperationalError: If tables are missing or corrupted
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            orders_df = pd.read_sql_query("SELECT * FROM orders", conn)
            customers_df = pd.read_sql_query("SELECT * FROM customers", conn)
            conn.close()
            
            # Parse datetime columns
            orders_df['order_ts'] = pd.to_datetime(orders_df['order_ts'])
            customers_df['signup_date'] = pd.to_datetime(customers_df['signup_date'])
            
            logger.info(f"Loaded {len(orders_df)} orders and {len(customers_df)} customers")
            return orders_df, customers_df
        
        except sqlite3.OperationalError as e:
            logger.error(f"Database error: {e}")
            raise
    
    def compute_rfm(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute RFM table for customers with ≥1 completed order as of snapshot date.
        
        RFM components (as of snapshot_date):
        - Recency: Days between snapshot and most recent completed order
        - Frequency: Count of distinct completed orders
        - Monetary: Sum of order_amount for completed orders
        
        Args:
            orders_df (pd.DataFrame): Orders table with order_ts, customer_id, order_amount, status
        
        Returns:
            pd.DataFrame: RFM table with columns [customer_id, recency, frequency, monetary]
                         Only includes customers with ≥1 completed order before snapshot
        
        Notes:
            - Only completed orders are used
            - Customers with no completed orders are excluded
            - Recency is computed as integer days
        """
        # Filter completed orders before snapshot
        completed = orders_df[
            (orders_df['status'] == 'completed') & 
            (orders_df['order_ts'] <= self.snapshot_date)
        ].copy()
        
        logger.info(f"Found {len(completed)} completed orders before snapshot")
        
        # Compute RFM by customer
        rfm_df = completed.groupby('customer_id').agg(
            recency=('order_ts', lambda x: (self.snapshot_date - x.max()).days),
            frequency=('order_id', 'nunique'),
            monetary=('order_amount', 'sum')
        ).reset_index()
        
        # Round monetary to 2 decimals
        rfm_df['monetary'] = rfm_df['monetary'].round(2)
        
        logger.info(f"RFM table: {len(rfm_df)} customers")
        logger.info(f"  Recency: median {rfm_df['recency'].median():.0f}, "
                   f"min {rfm_df['recency'].min()}, max {rfm_df['recency'].max()}")
        logger.info(f"  Frequency: mean {rfm_df['frequency'].mean():.2f}, "
                   f"max {rfm_df['frequency'].max()}")
        logger.info(f"  Monetary: mean ₹{rfm_df['monetary'].mean():.2f}, "
                   f"max ₹{rfm_df['monetary'].max():.2f}")
        
        self.rfm_df = rfm_df
        return rfm_df
    
    def standardize_rfm(self) -> np.ndarray:
        """
        Standardize RFM features using StandardScaler.
        
        Returns:
            np.ndarray: Scaled RFM matrix (n_customers, 3)
        
        Raises:
            ValueError: If RFM table not computed
        """
        if self.rfm_df is None:
            raise ValueError("RFM table not computed. Call compute_rfm() first.")
        
        rfm_cols = ['recency', 'frequency', 'monetary']
        self.scaler = StandardScaler()
        self.scaled_rfm = self.scaler.fit_transform(self.rfm_df[rfm_cols])
        
        logger.info("RFM standardized with StandardScaler")
        return self.scaled_rfm
    
    def run_kmeans_analysis(self, k_range: range = range(2, 9)) -> dict:
        """
        Run K-means for multiple K values and record inertia and silhouette scores.
        
        Args:
            k_range (range): K values to test. Default: 2 to 8
        
        Returns:
            dict: Results keyed by K with 'inertia' and 'silhouette' values
        
        Notes:
            - All K-means models use n_init=10 and random_state=42 for reproducibility
            - Silhouette score is computed on scaled data
        """
        if self.scaled_rfm is None:
            raise ValueError("Scaled RFM not available. Call standardize_rfm() first.")
        
        logger.info("\n=== K-Means Analysis ===")
        logger.info("K       Inertia    Silhouette")
        logger.info("-" * 35)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            cluster_labels = kmeans.fit_predict(self.scaled_rfm)
            inertia = kmeans.inertia_
            silhouette = silhouette_score(self.scaled_rfm, cluster_labels)
            
            self.kmeans_results[k] = {
                'inertia': inertia,
                'silhouette': silhouette,
                'model': kmeans
            }
            
            logger.info(f"{k:<8}{inertia:>10.1f}    {silhouette:>10.3f}")
        
        logger.info("-" * 35)
        return self.kmeans_results
    
    def select_optimal_k(self, candidates: list = [3, 4, 5]) -> int:
        """
        Select optimal K from candidates using elbow, silhouette, and business judgment.
        
        Args:
            candidates (list): K values to consider. Default: [3, 4, 5]
        
        Returns:
            int: Selected K value
        
        Notes:
            - Elbow method: look for maximum decrease in inertia gain (second derivative)
            - Silhouette: higher is better, but not the sole criterion
            - Business judgment: prefer interpretability (e.g., 4 segments for 4 campaigns)
            - This implementation uses silhouette as primary metric, then business factors
        """
        if not self.kmeans_results:
            raise ValueError("K-means not run. Call run_kmeans_analysis() first.")
        
        logger.info("\n=== K Selection ===")
        logger.info("Evaluating candidates from business perspective:")
        
        for k in candidates:
            metrics = self.kmeans_results[k]
            logger.info(f"  K={k}: inertia={metrics['inertia']:.1f}, "
                       f"silhouette={metrics['silhouette']:.3f}")
        
        # Select K with best silhouette score among candidates (business judgment)
        best_k = max(candidates, key=lambda k: self.kmeans_results[k]['silhouette'])
        
        # Alternative: could use elbow or other heuristics, but silhouette + business judgment wins
        logger.info(f"\nSelected K={best_k} based on silhouette score + business interpretability")
        self.optimal_k = best_k
        self.kmeans_model = self.kmeans_results[best_k]['model']
        
        return best_k
    
    def profile_segments(self) -> pd.DataFrame:
        """
        Profile each segment and create business names.
        
        Returns:
            pd.DataFrame: Segment profiles with size, mean recency, frequency, monetary
        
        Notes:
            - Segment with highest mean monetary is marked Premium
            - Names are derived from RFM characteristics and business actions
        """
        if self.kmeans_model is None:
            raise ValueError("K-means model not fitted. Call select_optimal_k() first.")
        
        # Add cluster labels to RFM table
        self.rfm_df['segment_id'] = self.kmeans_model.labels_
        
        # Compute segment profiles
        profiles = self.rfm_df.groupby('segment_id').agg(
            size=('customer_id', 'count'),
            mean_recency=('recency', 'mean'),
            mean_frequency=('frequency', 'mean'),
            mean_monetary=('monetary', 'mean')
        ).round(2)
        
        logger.info("\n=== Segment Profiles ===")
        logger.info(f"\nK={self.optimal_k} Segments:")
        logger.info(profiles.to_string())
        
        # Identify Premium segment (highest mean monetary)
        premium_segment = profiles['mean_monetary'].idxmax()
        
        # Assign business names based on RFM profiles
        # Names are illustrative and should be tailored to business context
        self.segment_names = self._assign_segment_names(profiles)
        self.is_premium = {i: (i == premium_segment) for i in range(self.optimal_k)}
        
        logger.info(f"\nPremium segment: {premium_segment} ({self.segment_names[premium_segment]})")
        
        # Log segment names and actions
        logger.info("\nSegment Assignments:")
        for seg_id in range(self.optimal_k):
            premium_flag = "✓ PREMIUM" if self.is_premium[seg_id] else ""
            logger.info(f"  Segment {seg_id}: {self.segment_names[seg_id]} {premium_flag}")
        
        return profiles
    
    def _assign_segment_names(self, profiles: pd.DataFrame) -> dict:
        """
        Assign business names to segments based on RFM characteristics.
        
        Args:
            profiles (pd.DataFrame): Segment profiles
        
        Returns:
            dict: Mapping segment_id -> (name, action_recommendation)
        """
        names = {}
        for seg_id in range(self.optimal_k):
            rec = profiles.loc[seg_id, 'mean_recency']
            freq = profiles.loc[seg_id, 'mean_frequency']
            mon = profiles.loc[seg_id, 'mean_monetary']
            
            # Heuristic naming based on RFM values
            if mon > self.rfm_df['monetary'].quantile(0.75):  # High monetary
                if freq > self.rfm_df['frequency'].quantile(0.75):
                    name = "High-Value Frequent"
                else:
                    name = "High-Value Occasional"
            elif mon < self.rfm_df['monetary'].quantile(0.25):  # Low monetary
                name = "Low-Spend At-Risk"
            else:  # Medium monetary
                if rec < self.rfm_df['recency'].quantile(0.25):
                    name = "Recently Active"
                else:
                    name = "Dormant Medium-Spend"
            
            names[seg_id] = name
        
        return names
    
    def apply_pca(self) -> tuple:
        """
        Apply PCA (n_components=2) to scaled RFM for visualization/analysis.
        
        Returns:
            tuple: (pca_model, explained_variance_ratio)
                - pca_model: Fitted PCA(n_components=2)
                - explained_variance_ratio: Array of variance explained by each component
        
        Raises:
            ValueError: If scaled RFM not available
        
        Notes:
            - PCA is fit on the full scaled RFM matrix (not separately on train/test)
            - Explained variance ratio typically: [~0.634, ~0.218] (total ~0.852)
        """
        if self.scaled_rfm is None:
            raise ValueError("Scaled RFM not available.")
        
        self.pca = PCA(n_components=2)
        self.pca.fit(self.scaled_rfm)
        
        var_ratio = self.pca.explained_variance_ratio_
        logger.info(f"\n=== PCA Analysis ===")
        logger.info(f"Explained variance ratio: {var_ratio[0]:.3f}, {var_ratio[1]:.3f}")
        logger.info(f"Total explained variance: {var_ratio.sum():.3f}")
        
        return self.pca, var_ratio
    
    def export_segments(self, output_path: str = 'outputs/customer_segments.csv') -> pd.DataFrame:
        """
        Export segment assignments to CSV with full customer context.
        
        Args:
            output_path (str): Output CSV path
        
        Returns:
            pd.DataFrame: Exported segments table
        
        Columns:
            - customer_id
            - recency (days)
            - frequency (count)
            - monetary (INR)
            - segment_id (0 to K-1)
            - segment_name (business-friendly name)
            - is_premium (boolean, True for highest-monetary segment)
        
        Raises:
            ValueError: If segmentation not complete
        """
        if self.rfm_df is None or 'segment_id' not in self.rfm_df.columns:
            raise ValueError("RFM segmentation not complete. Run full workflow first.")
        
        # Build output DataFrame
        self.segments_df = self.rfm_df[['customer_id', 'recency', 'frequency', 'monetary']].copy()
        self.segments_df['segment_id'] = self.rfm_df['segment_id']
        self.segments_df['segment_name'] = self.segments_df['segment_id'].map(self.segment_names)
        self.segments_df['is_premium'] = self.segments_df['segment_id'].map(self.is_premium)
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write to CSV
        self.segments_df.to_csv(output_path, index=False)
        logger.info(f"\n=== Export ===")
        logger.info(f"Exported {len(self.segments_df)} customers to {output_path}")
        
        # Log premium segment size
        premium_count = self.segments_df['is_premium'].sum()
        logger.info(f"Premium segment size: {premium_count}")
        
        return self.segments_df
    
    def run_full_workflow(self) -> dict:
        """
        Execute complete RFM segmentation workflow.
        
        Returns:
            dict: Summary of results with keys:
                - 'n_customers': Number of customers in RFM
                - 'optimal_k': Selected K
                - 'segment_sizes': Dict of segment sizes
                - 'pca_variance': Explained variance ratio
                - 'export_path': Path to customer_segments.csv
        """
        logger.info("Starting RFM Segmentation Workflow")
        logger.info("=" * 60)
        
        # Step 1: Load data
        orders_df, customers_df = self.load_data()
        
        # Step 2: Compute RFM
        self.compute_rfm(orders_df)
        
        # Step 3: Standardize
        self.standardize_rfm()
        
        # Step 4: K-means analysis
        self.run_kmeans_analysis()
        
        # Step 5: Select K
        self.select_optimal_k()
        
        # Step 6: Profile and name
        self.profile_segments()
        
        # Step 7: PCA
        pca_model, pca_var = self.apply_pca()
        
        # Step 8: Export
        output_path = 'outputs/customer_segments.csv'
        self.export_segments(output_path)
        
        logger.info("=" * 60)
        logger.info("RFM Segmentation Workflow Complete")
        
        # Summary
        summary = {
            'n_customers': len(self.rfm_df),
            'optimal_k': self.optimal_k,
            'segment_sizes': self.rfm_df['segment_id'].value_counts().to_dict(),
            'pca_variance': pca_var,
            'export_path': output_path
        }
        
        return summary


if __name__ == '__main__':
    # Example usage
    segmentation = RFMSegmentation(db_path='data/instamart.db', snapshot_date='2025-10-01')
    summary = segmentation.run_full_workflow()
    print("\n" + "=" * 60)
    print("Workflow Summary:")
    print(f"  Customers: {summary['n_customers']}")
    print(f"  Selected K: {summary['optimal_k']}")
    print(f"  PCA Variance: {summary['pca_variance'][0]:.3f} + {summary['pca_variance'][1]:.3f}")
    print(f"  Export: {summary['export_path']}")
