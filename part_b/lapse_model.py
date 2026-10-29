"""
Part B Task 4: Lapse Prediction Model

This module builds a lapse prediction model to identify high-risk customers.
A customer is labeled as "lapsed" if they place zero completed orders in the 
30-day window [2025-09-01, 2025-09-30], given that they had activity before 2025-09-01.

Workflow:
  1. Load orders, customers, RFM segments from Part A/B
  2. Define lapse cohort: customers with ≥1 completed order before 2025-09-01
  3. Compute lapse label: check for completed orders in [2025-09-01, 2025-09-30]
  4. Engineer features (13+):
     - RFM: recency, frequency, monetary (pre-cutoff orders)
     - Behavioral: avg_discount, cancellation_rate, avg_delivery_minutes
     - Categorical: top_category, city, gender
     - Customer: age, tenure_days
  5. Split into train/test (80/20 stratified)
  6. Preprocess:
     - OneHotEncoder(drop="first"): city, gender, top_category
     - StandardScaler: recency, frequency, monetary, numeric features
     - Pipelines fitted on training only, applied to test (no data leakage)
  7. Train models with GridSearchCV:
     - LogisticRegression: C ∈ {0.01, 0.1, 1, 10, 100}, cv=5, scoring="f1"
     - RandomForestClassifier: max_depth ∈ {3,5,8}, min_samples_leaf ∈ {5,10,20}, cv=5
  8. Class balance experiment: oversample minority (lapsed) class in training, refit LogReg
  9. Persist models to joblib
  10. Export test predictions with lapse probability

Reference Checkpoints:
  - Cohort size: 1,156 customers (with pre-cutoff activity)
  - Lapsed count: 443 (38.32% lapse rate)
  - Train/test split: 924 / 232 (stratified, random_state=42)
  - Baseline accuracy (predict all active): 61.6%
  - Target test F1 score: ≥0.55
  - Class balance (training): ~38% lapsed (before oversampling)

Safeguards:
  - Features use only orders before 2025-09-01 (no test data leakage)
  - RFM segment labels from File 1 NOT used as features (task 3 outputs excluded)
  - Preprocessing pipeline fitted on training only
  - All random seeds fixed: random_state=42
  - Reproducible train/test split via stratification
"""

import sqlite3
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LapseModel:
    """
    Build and evaluate a lapse prediction model.
    
    A customer is lapsed if they place zero completed orders in [2025-09-01, 2025-09-30].
    Training uses only orders before 2025-09-01 to ensure no data leakage.
    """

    def __init__(self, db_path='data/instamart.db', 
                 cohort_cutoff='2025-09-01', 
                 lapse_window_start='2025-09-01',
                 lapse_window_end='2025-09-30'):
        """
        Initialize LapseModel.
        
        Parameters
        ----------
        db_path : str
            Path to SQLite database with orders and customers tables
        cohort_cutoff : str
            Cutoff date for cohort definition (YYYY-MM-DD).
            Customers with ≥1 completed order before this date form the cohort.
        lapse_window_start : str
            Start of lapse observation window (YYYY-MM-DD)
        lapse_window_end : str
            End of lapse observation window (YYYY-MM-DD)
        """
        self.db_path = db_path
        self.cohort_cutoff = pd.to_datetime(cohort_cutoff)
        self.lapse_window_start = pd.to_datetime(lapse_window_start)
        self.lapse_window_end = pd.to_datetime(lapse_window_end)
        
        self.orders_df = None
        self.customers_df = None
        self.features_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_indices = None
        self.test_indices = None
        
        self.preprocessor = None
        self.logreg_model = None
        self.rf_model = None
        self.logreg_balanced_model = None
        
        logger.info(f"Initialized LapseModel with cohort_cutoff={cohort_cutoff}")

    def load_data(self):
        """
        Load orders and customers from SQLite database.
        Parse datetimes for all date columns.
        
        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame)
            orders_df, customers_df
        """
        conn = sqlite3.connect(self.db_path)
        orders_df = pd.read_sql_query("SELECT * FROM orders", conn)
        customers_df = pd.read_sql_query("SELECT * FROM customers", conn)
        conn.close()
        
        # Parse datetimes
        orders_df['order_ts'] = pd.to_datetime(orders_df['order_ts'])
        customers_df['signup_date'] = pd.to_datetime(customers_df['signup_date'])
        
        self.orders_df = orders_df
        self.customers_df = customers_df
        
        logger.info(f"Loaded {len(orders_df)} orders and {len(customers_df)} customers")
        return orders_df, customers_df

    def define_lapse_cohort(self):
        """
        Define cohort as customers with ≥1 completed order before cohort_cutoff.
        
        Returns
        -------
        list
            customer_ids in cohort
        """
        pre_cutoff = self.orders_df[
            (self.orders_df['order_ts'] < self.cohort_cutoff) &
            (self.orders_df['status'] == 'completed')
        ]
        cohort = pre_cutoff['customer_id'].unique().tolist()
        logger.info(f"Cohort size: {len(cohort)} customers with pre-cutoff activity")
        return cohort

    def compute_lapse_label(self, cohort):
        """
        Compute lapse label for cohort customers.
        
        A customer is lapsed (1) if they have zero completed orders in 
        [lapse_window_start, lapse_window_end]. Non-lapsed (0) otherwise.
        
        Parameters
        ----------
        cohort : list
            customer_ids to label
        
        Returns
        -------
        pd.Series
            lapse label indexed by customer_id
        """
        # Orders in lapse window
        lapse_window_orders = self.orders_df[
            (self.orders_df['order_ts'] >= self.lapse_window_start) &
            (self.orders_df['order_ts'] <= self.lapse_window_end) &
            (self.orders_df['status'] == 'completed')
        ]
        
        # Customers with ≥1 order in lapse window are not lapsed
        active_in_window = set(lapse_window_orders['customer_id'].unique())
        
        # Create label: 1 = lapsed, 0 = active
        labels = []
        for cid in cohort:
            labels.append(1 if cid not in active_in_window else 0)
        
        lapse_series = pd.Series(labels, index=cohort, name='is_lapsed')
        lapse_count = (lapse_series == 1).sum()
        lapse_rate = lapse_count / len(cohort) * 100
        
        logger.info(f"Lapse label computed: {lapse_count} lapsed ({lapse_rate:.2f}%) out of {len(cohort)}")
        return lapse_series

    def engineer_features(self, cohort):
        """
        Engineer 13+ features for lapse prediction using pre-cutoff data.
        
        Features:
          RFM (pre-cutoff):
            - recency: days from last completed order to cohort_cutoff
            - frequency: count of completed orders before cohort_cutoff
            - monetary: sum of order_amount for completed orders before cohort_cutoff
          Behavioral:
            - avg_discount: mean discount_pct (pre-cutoff completed orders)
            - cancellation_rate: frac of cancelled orders (pre-cutoff all orders)
            - avg_delivery_minutes: mean delivery_minutes (pre-cutoff, non-null)
          Categorical:
            - top_category: most frequent product_id prefix (e.g. 'P01') in pre-cutoff orders
            - city: from customers table
            - gender: from customers table
          Customer:
            - age: from customers table
            - tenure_days: days from signup_date to cohort_cutoff
        
        Parameters
        ----------
        cohort : list
            customer_ids to engineer features for
        
        Returns
        -------
        pd.DataFrame
            Features indexed by customer_id
        """
        # Pre-cutoff orders only
        pre_cutoff_orders = self.orders_df[
            (self.orders_df['order_ts'] < self.cohort_cutoff) &
            (self.orders_df['customer_id'].isin(cohort))
        ]
        
        features = pd.DataFrame(index=cohort)
        features.index.name = 'customer_id'
        
        # RFM features (completed orders only)
        completed_orders = pre_cutoff_orders[pre_cutoff_orders['status'] == 'completed']
        
        # Recency
        last_order_date = completed_orders.groupby('customer_id')['order_ts'].max()
        recency = (self.cohort_cutoff - last_order_date).dt.days
        features['recency'] = recency
        
        # Frequency
        frequency = completed_orders.groupby('customer_id').size()
        features['frequency'] = frequency
        
        # Monetary
        monetary = completed_orders.groupby('customer_id')['order_amount'].sum()
        features['monetary'] = monetary
        
        # Behavioral features
        # avg_discount (completed orders)
        avg_discount = completed_orders.groupby('customer_id')['discount_pct'].mean()
        features['avg_discount'] = avg_discount
        
        # cancellation_rate (all pre-cutoff orders)
        total_orders = pre_cutoff_orders.groupby('customer_id').size()
        cancelled_orders = pre_cutoff_orders[
            pre_cutoff_orders['status'] == 'cancelled'
        ].groupby('customer_id').size()
        cancellation_rate = cancelled_orders / total_orders
        features['cancellation_rate'] = cancellation_rate
        
        # avg_delivery_minutes (completed orders with non-null delivery)
        delivery_times = completed_orders.dropna(subset=['delivery_minutes'])
        avg_delivery = delivery_times.groupby('customer_id')['delivery_minutes'].mean()
        features['avg_delivery_minutes'] = avg_delivery
        
        # top_category: most frequent product_id (e.g. 'P001' -> 'P00')
        # For simplicity, use product_id as category
        def get_top_category(cid):
            cust_orders = pre_cutoff_orders[pre_cutoff_orders['customer_id'] == cid]
            if len(cust_orders) == 0:
                return 'unknown'
            top_prod = cust_orders['product_id'].mode()
            if len(top_prod) > 0:
                return top_prod.iloc[0][:3]  # e.g., 'P01' from 'P001'
            return 'unknown'
        
        features['top_category'] = features.index.map(get_top_category)
        
        # Categorical features from customers table
        cust_lookup = self.customers_df.set_index('customer_id')
        features['city'] = features.index.map(lambda x: cust_lookup.loc[x, 'city'] if x in cust_lookup.index else 'unknown')
        features['gender'] = features.index.map(lambda x: cust_lookup.loc[x, 'gender'] if x in cust_lookup.index else 'unknown')
        
        # Customer features
        features['age'] = features.index.map(lambda x: cust_lookup.loc[x, 'age'] if x in cust_lookup.index else features['age'].median())
        features['tenure_days'] = features.index.map(
            lambda x: (self.cohort_cutoff - cust_lookup.loc[x, 'signup_date']).days if x in cust_lookup.index else 0
        )
        
        # Fill NaNs for numeric features with median
        numeric_cols = ['recency', 'frequency', 'monetary', 'avg_discount', 
                       'cancellation_rate', 'avg_delivery_minutes', 'age', 'tenure_days']
        for col in numeric_cols:
            features[col] = features[col].fillna(features[col].median())
        
        # Fill categorical NaNs
        features['top_category'] = features['top_category'].fillna('unknown')
        features['city'] = features['city'].fillna('unknown')
        features['gender'] = features['gender'].fillna('unknown')
        
        self.features_df = features
        logger.info(f"Engineered features for {len(features)} customers ({len(features.columns)} features)")
        logger.info(f"Columns: {', '.join(features.columns.tolist())}")
        
        return features

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split features and labels into train/test sets using stratified split.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Labels
        test_size : float
            Test set fraction
        random_state : int
            Random seed for reproducibility
        
        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test, train_idx, test_idx)
        """
        train_idx, test_idx = train_test_split(
            range(len(X)),
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )
        
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_indices = train_idx
        self.test_indices = test_idx
        
        logger.info(f"Train/test split: {len(X_train)} / {len(X_test)} (stratified, random_state={random_state})")
        logger.info(f"Train lapse rate: {(y_train == 1).sum() / len(y_train) * 100:.2f}%")
        logger.info(f"Test lapse rate: {(y_test == 1).sum() / len(y_test) * 100:.2f}%")
        
        return X_train, X_test, y_train, y_test, train_idx, test_idx

    def build_preprocessor(self):
        """
        Build preprocessing pipeline with OneHotEncoder for categorical features
        and StandardScaler for numeric features.
        
        Fitted on training data only (no leakage).
        
        Returns
        -------
        sklearn.compose.ColumnTransformer
            Preprocessor fitted on training data
        """
        numeric_features = ['recency', 'frequency', 'monetary', 'avg_discount',
                           'cancellation_rate', 'avg_delivery_minutes', 'age', 'tenure_days']
        categorical_features = ['city', 'gender', 'top_category']
        
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Fit on training data only
        preprocessor.fit(self.X_train)
        
        self.preprocessor = preprocessor
        logger.info("Preprocessor built and fitted on training data")
        logger.info(f"Numeric features: {numeric_features}")
        logger.info(f"Categorical features: {categorical_features}")
        
        return preprocessor

    def train_logistic_regression(self, use_balanced=False):
        """
        Train LogisticRegression with GridSearchCV.
        
        Hyperparameters: C ∈ {0.01, 0.1, 1, 10, 100}
        Cross-validation: 5-fold stratified
        Scoring: F1 (macro)
        
        Parameters
        ----------
        use_balanced : bool
            If True, use RandomOverSampler to balance minority class in training
        
        Returns
        -------
        sklearn.linear_model.LogisticRegression
            Fitted model (best estimator from GridSearchCV)
        """
        X_train_prep = self.preprocessor.transform(self.X_train)
        
        if use_balanced:
            # Oversample minority class (lapsed)
            ros = RandomOverSampler(random_state=42)
            X_train_balanced, y_train_balanced = ros.fit_resample(X_train_prep, self.y_train)
            logger.info(f"Balanced training: {len(X_train_balanced)} samples (oversampled)")
        else:
            X_train_balanced = X_train_prep
            y_train_balanced = self.y_train
        
        # GridSearchCV for LogisticRegression
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
        
        grid = GridSearchCV(
            LogisticRegression(max_iter=1000, random_state=42),
            param_grid,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1
        )
        
        grid.fit(X_train_balanced, y_train_balanced)
        
        best_model = grid.best_estimator_
        
        if use_balanced:
            self.logreg_balanced_model = best_model
            logger.info(f"Balanced LogisticRegression trained: best C={grid.best_params_['C']}")
        else:
            self.logreg_model = best_model
            logger.info(f"LogisticRegression trained: best C={grid.best_params_['C']}")
            logger.info(f"Cross-val score (F1): {grid.best_score_:.4f}")
        
        return best_model

    def train_random_forest(self):
        """
        Train RandomForestClassifier with GridSearchCV.
        
        Hyperparameters: 
          max_depth ∈ {3, 5, 8}
          min_samples_leaf ∈ {5, 10, 20}
        Cross-validation: 5-fold stratified
        Scoring: F1 (macro)
        
        Returns
        -------
        sklearn.ensemble.RandomForestClassifier
            Fitted model (best estimator from GridSearchCV)
        """
        X_train_prep = self.preprocessor.transform(self.X_train)
        
        param_grid = {
            'max_depth': [3, 5, 8],
            'min_samples_leaf': [5, 10, 20]
        }
        
        grid = GridSearchCV(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            param_grid,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1
        )
        
        grid.fit(X_train_prep, self.y_train)
        
        best_model = grid.best_estimator_
        self.rf_model = best_model
        
        logger.info(f"RandomForest trained: best params={grid.best_params_}")
        logger.info(f"Cross-val score (F1): {grid.best_score_:.4f}")
        
        return best_model

    def evaluate_model(self, model, model_name='Model'):
        """
        Evaluate model on test set and print metrics.
        
        Parameters
        ----------
        model : sklearn estimator
            Fitted model
        model_name : str
            Name for logging
        
        Returns
        -------
        dict
            Metrics: accuracy, precision, recall, f1, roc_auc, confusion_matrix
        """
        X_test_prep = self.preprocessor.transform(self.X_test)
        y_pred = model.predict(X_test_prep)
        y_pred_proba = model.predict_proba(X_test_prep)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
        
        logger.info(f"\n{model_name} Test Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
        
        return metrics

    def save_models(self, output_dir='outputs'):
        """
        Save trained models to joblib.
        
        Parameters
        ----------
        output_dir : str
            Directory to save models
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        if self.logreg_model:
            joblib.dump(self.logreg_model, f'{output_dir}/lapse_logreg.joblib')
            logger.info(f"Saved LogisticRegression to {output_dir}/lapse_logreg.joblib")
        
        if self.rf_model:
            joblib.dump(self.rf_model, f'{output_dir}/lapse_rf.joblib')
            logger.info(f"Saved RandomForest to {output_dir}/lapse_rf.joblib")
        
        if self.logreg_balanced_model:
            joblib.dump(self.logreg_balanced_model, f'{output_dir}/lapse_logreg_balanced.joblib')
            logger.info(f"Saved Balanced LogisticRegression to {output_dir}/lapse_logreg_balanced.joblib")
        
        if self.preprocessor:
            joblib.dump(self.preprocessor, f'{output_dir}/lapse_preprocessor.joblib')
            logger.info(f"Saved preprocessor to {output_dir}/lapse_preprocessor.joblib")

    def export_predictions(self, model, output_path='outputs/lapse_predictions.csv'):
        """
        Export test predictions with customer_id, actual label, predicted label, and probability.
        
        Parameters
        ----------
        model : sklearn estimator
            Fitted model
        output_path : str
            Path to save predictions CSV
        """
        Path(output_path).parent.mkdir(exist_ok=True)
        
        X_test_prep = self.preprocessor.transform(self.X_test)
        y_pred = model.predict(X_test_prep)
        y_pred_proba = model.predict_proba(X_test_prep)[:, 1]
        
        # Get customer_ids for test set
        test_customer_ids = self.features_df.index[self.test_indices].tolist()
        
        predictions_df = pd.DataFrame({
            'customer_id': test_customer_ids,
            'actual_lapsed': self.y_test.values,
            'predicted_lapsed': y_pred,
            'lapse_probability': y_pred_proba
        })
        
        predictions_df.to_csv(output_path, index=False)
        logger.info(f"Exported predictions to {output_path} ({len(predictions_df)} rows)")
        
        return predictions_df

    def run_full_workflow(self, output_dir='outputs'):
        """
        Execute complete lapse modeling pipeline.
        
        Returns
        -------
        dict
            Workflow summary with cohort size, lapse count, model metrics
        """
        logger.info("Starting Lapse Prediction Workflow")
        logger.info("=" * 60)
        
        # 1. Load data
        self.load_data()
        
        # 2. Define cohort and lapse label
        cohort = self.define_lapse_cohort()
        lapse_label = self.compute_lapse_label(cohort)
        
        # 3. Engineer features
        features = self.engineer_features(cohort)
        
        # 4. Split data
        self.split_data(features, lapse_label)
        
        # 5. Build preprocessor
        self.build_preprocessor()
        
        # 6. Train models
        logger.info("\nTraining LogisticRegression...")
        self.train_logistic_regression(use_balanced=False)
        
        logger.info("\nTraining RandomForestClassifier...")
        self.train_random_forest()
        
        logger.info("\nTraining Balanced LogisticRegression...")
        self.train_logistic_regression(use_balanced=True)
        
        # 7. Evaluate models
        logger.info("\n" + "=" * 60)
        logreg_metrics = self.evaluate_model(self.logreg_model, "LogisticRegression")
        rf_metrics = self.evaluate_model(self.rf_model, "RandomForest")
        logreg_balanced_metrics = self.evaluate_model(self.logreg_balanced_model, "Balanced LogisticRegression")
        
        # 8. Save models
        self.save_models(output_dir)
        
        # 9. Export predictions
        self.export_predictions(self.logreg_model, f'{output_dir}/lapse_predictions_logreg.csv')
        self.export_predictions(self.rf_model, f'{output_dir}/lapse_predictions_rf.csv')
        self.export_predictions(self.logreg_balanced_model, f'{output_dir}/lapse_predictions_balanced.csv')
        
        logger.info("=" * 60)
        logger.info("Lapse Prediction Workflow Complete")
        
        summary = {
            'cohort_size': len(cohort),
            'lapse_count': (lapse_label == 1).sum(),
            'lapse_rate': (lapse_label == 1).sum() / len(cohort),
            'train_size': len(self.X_train),
            'test_size': len(self.X_test),
            'logreg_f1': logreg_metrics['f1'],
            'rf_f1': rf_metrics['f1'],
            'logreg_balanced_f1': logreg_balanced_metrics['f1'],
        }
        
        return summary
