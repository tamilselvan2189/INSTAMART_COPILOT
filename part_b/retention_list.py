"""
Part B Task 5: Retention List Generation

This module generates a prioritized retention list targeting high-risk Premium customers
who are predicted to lapse.

Workflow:
  1. Load test predictions from File 2 (lapse model outputs)
  2. Load segment assignments from File 1 (RFM segmentation)
  3. Filter Premium customers from test set
  4. Rank by lapse probability (descending)
  5. Export top 10 to CSV with customer_id, lapse_probability, actually_lapsed
  6. Report count of truly lapsed in top 10

Reference Checkpoints:
  - Premium segment size (full cohort): 138 customers
  - Premium customers in test set: typically 10-12 customers
  - Top 10 from retention list: top 10 highest lapse probabilities among Premium in test
  - Illustrative metric: count of actually_lapsed=1 among top 10

Safeguards & Limitations:
  - Premium flag is post-hoc (computed from whole-window RFM), not predictive
  - Retention list is illustrative; Premium label uses data from entire observation window
  - Lapse predictions use pre-cutoff features only (no data leakage into features)
  - Premium classification uses entire RFM window (post-hoc, not leaked)
  - Production use: Premium should be defined pre-cutoff only

Data Flow:
  - Input 1: outputs/lapse_predictions_*.csv from File 2
  - Input 2: outputs/customer_segments.csv from File 1
  - Output: outputs/retention_list.csv
"""

import pandas as pd
import joblib
from pathlib import Path
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetentionListGenerator:
    """
    Generate a prioritized retention list for high-risk Premium customers.
    """

    def __init__(self, segments_path='outputs/customer_segments.csv',
                 predictions_path='outputs/lapse_predictions_logreg.csv',
                 top_n=10):
        """
        Initialize RetentionListGenerator.
        
        Parameters
        ----------
        segments_path : str
            Path to customer_segments.csv from File 1 (RFM segmentation)
        predictions_path : str
            Path to lapse predictions CSV from File 2
            Should contain columns: customer_id, actual_lapsed, predicted_lapsed, lapse_probability
        top_n : int
            Number of top customers to include in retention list
        """
        self.segments_path = segments_path
        self.predictions_path = predictions_path
        self.top_n = top_n
        
        self.segments_df = None
        self.predictions_df = None
        self.retention_list = None
        self.premium_count = None
        self.retention_count = None
        
        logger.info(f"Initialized RetentionListGenerator with top_n={top_n}")

    def load_data(self):
        """
        Load segment assignments and predictions.
        
        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame)
            segments_df, predictions_df
        """
        # Load segments from File 1
        segments_df = pd.read_csv(self.segments_path)
        logger.info(f"Loaded {len(segments_df)} customer segments from {self.segments_path}")
        
        # Load predictions from File 2
        predictions_df = pd.read_csv(self.predictions_path)
        logger.info(f"Loaded {len(predictions_df)} predictions from {self.predictions_path}")
        
        self.segments_df = segments_df
        self.predictions_df = predictions_df
        
        return segments_df, predictions_df

    def filter_premium(self):
        """
        Filter Premium segment customers from predictions.
        
        Returns
        -------
        pd.DataFrame
            Premium customers with their lapse predictions
        """
        # Get Premium customer IDs
        premium_ids = set(self.segments_df[self.segments_df['is_premium'] == True]['customer_id'])
        self.premium_count = len(premium_ids)
        
        logger.info(f"Premium segment size (full cohort): {self.premium_count}")
        
        # Filter predictions to Premium customers only
        premium_predictions = self.predictions_df[
            self.predictions_df['customer_id'].isin(premium_ids)
        ].copy()
        
        logger.info(f"Premium customers in test set: {len(premium_predictions)}")
        
        return premium_predictions

    def rank_by_lapse_probability(self, premium_predictions):
        """
        Rank Premium customers by predicted lapse probability (descending).
        
        Parameters
        ----------
        premium_predictions : pd.DataFrame
            Premium customer predictions
        
        Returns
        -------
        pd.DataFrame
            Sorted by lapse_probability descending
        """
        ranked = premium_predictions.sort_values(
            'lapse_probability',
            ascending=False
        ).reset_index(drop=True)
        
        return ranked

    def generate_retention_list(self):
        """
        Generate the final retention list: top N Premium customers by lapse probability.
        
        Returns
        -------
        pd.DataFrame
            Retention list with columns: customer_id, lapse_probability, actually_lapsed
        """
        # Load and filter
        premium_predictions = self.filter_premium()
        
        # If no Premium customers in test set, return empty list
        if len(premium_predictions) == 0:
            logger.warning("No Premium customers in test set")
            self.retention_list = pd.DataFrame(
                columns=['customer_id', 'lapse_probability', 'actually_lapsed']
            )
            self.retention_count = 0
            return self.retention_list
        
        # Rank by lapse probability
        ranked = self.rank_by_lapse_probability(premium_predictions)
        
        # Take top N (or all if fewer than top_n)
        retention_list = ranked[['customer_id', 'lapse_probability', 'actual_lapsed']].head(self.top_n)
        retention_list.columns = ['customer_id', 'lapse_probability', 'actually_lapsed']
        
        self.retention_list = retention_list
        self.retention_count = len(retention_list)
        
        logger.info(f"Generated retention list: {self.retention_count} customers")
        
        return retention_list

    def report_statistics(self):
        """
        Compute and report statistics on retention list.
        
        Returns
        -------
        dict
            Statistics including count of truly lapsed in top N
        """
        if self.retention_list is None:
            logger.warning("No retention list generated yet")
            return {}
        
        truly_lapsed = (self.retention_list['actually_lapsed'] == 1).sum()
        truly_lapsed_pct = truly_lapsed / len(self.retention_list) * 100 if len(self.retention_list) > 0 else 0
        
        stats = {
            'premium_count_full': self.premium_count,
            'premium_count_test': len(self.predictions_df[
                self.predictions_df['customer_id'].isin(
                    set(self.segments_df[self.segments_df['is_premium'] == True]['customer_id'])
                )
            ]),
            'retention_list_size': self.retention_count,
            'truly_lapsed_in_retention': truly_lapsed,
            'truly_lapsed_pct': truly_lapsed_pct,
        }
        
        logger.info(f"\n=== Retention List Statistics ===")
        logger.info(f"Premium segment (full cohort): {stats['premium_count_full']}")
        logger.info(f"Premium in test set: {stats['premium_count_test']}")
        logger.info(f"Retention list size: {stats['retention_list_size']}")
        logger.info(f"Truly lapsed in retention list: {truly_lapsed} ({truly_lapsed_pct:.1f}%)")
        
        return stats

    def export_retention_list(self, output_path='outputs/retention_list.csv'):
        """
        Export retention list to CSV.
        
        Parameters
        ----------
        output_path : str
            Path to save retention list
        """
        Path(output_path).parent.mkdir(exist_ok=True)
        
        self.retention_list.to_csv(output_path, index=False)
        logger.info(f"Exported retention list to {output_path}")

    def run_full_workflow(self, output_path='outputs/retention_list.csv'):
        """
        Execute complete retention list generation pipeline.
        
        Parameters
        ----------
        output_path : str
            Path to save retention list
        
        Returns
        -------
        dict
            Workflow summary with statistics
        """
        logger.info("Starting Retention List Generation Workflow")
        logger.info("=" * 60)
        
        # 1. Load data
        self.load_data()
        
        # 2. Generate retention list
        retention_list = self.generate_retention_list()
        
        # 3. Report statistics
        stats = self.report_statistics()
        
        # 4. Export
        self.export_retention_list(output_path)
        
        logger.info("=" * 60)
        logger.info("Retention List Generation Workflow Complete")
        
        return stats
