"""
Data preprocessing utilities for BEED benchmark.
Handles data loading, cleaning, and feature scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles all data preprocessing operations."""
    
    def __init__(self, data_path: Path, processed_path: Path):
        self.data_path = data_path
        self.processed_path = processed_path
        self.scaler = StandardScaler()
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw dataset from CSV file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        logger.info(f"Loading raw data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        return df
    
    def preprocess_data(self, force_reprocess: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess data with caching.
        
        Args:
            force_reprocess: If True, reprocess even if cached data exists
            
        Returns:
            Tuple of (features, labels)
        """
        if self.processed_path.exists() and not force_reprocess:
            logger.info("Loading cached processed data")
            processed_df = pd.read_csv(self.processed_path)
            X = processed_df.drop('y', axis=1).values
            y = processed_df['y'].values
            return X, y
        
        logger.info("Processing raw data")
        df = self.load_raw_data()
        
        # Separate features and labels (assuming last column is target)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Normalize features
        logger.info("Applying StandardScaler normalization")
        X_scaled = self.scaler.fit_transform(X)
        
        # Save processed data
        self._save_processed_data(X_scaled, y, df.columns)
        
        return X_scaled, y
    
    def _save_processed_data(self, X: np.ndarray, y: np.ndarray, columns: pd.Index):
        """Save processed data to CSV for future use."""
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        
        processed_df = pd.DataFrame(X, columns=columns[:-1])
        processed_df['y'] = y
        processed_df.to_csv(self.processed_path, index=False)
        logger.info(f"Processed data saved to {self.processed_path}")
    
    def get_data_summary(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Generate data summary statistics."""
        return {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'class_distribution': pd.Series(y).value_counts().to_dict(),
            'feature_stats': {
                'mean_range': (X.mean(axis=0).min(), X.mean(axis=0).max()),
                'std_range': (X.std(axis=0).min(), X.std(axis=0).max())
            }
        }
