"""
Comprehensive model evaluation utilities.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
import time
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation with cross-validation."""
    
    def __init__(self, cv_folds: int = 10, random_state: int = 42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                                  random_state=random_state)
    
    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Perform comprehensive model evaluation using stratified cross-validation.
        
        Args:
            model: Scikit-learn compatible model
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary containing mean and std for each metric
        """
        fold_metrics = self._initialize_metrics()
        
        for fold, (train_idx, test_idx) in enumerate(self.skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Training time
            start_time = time.time()
            model.fit(X_train, y_train)
            fold_metrics['train_time'].append(time.time() - start_time)
            
            # Inference time
            start_time = time.time()
            y_pred = model.predict(X_test)
            fold_metrics['inference_time'].append(time.time() - start_time)
            
            # Calculate metrics
            self._calculate_fold_metrics(y_test, y_pred, model, X_test, fold_metrics)
        
        return self._aggregate_metrics(fold_metrics)
    
    def _initialize_metrics(self) -> Dict[str, List]:
        """Initialize metric storage for cross-validation folds."""
        return {
            'accuracy': [], 'precision_macro': [], 'recall_macro': [], 'f1_macro': [],
            'precision_weighted': [], 'recall_weighted': [], 'f1_weighted': [],
            'roc_auc_ovr': [], 'train_time': [], 'inference_time': []
        }
    
    def _calculate_fold_metrics(self, y_true, y_pred, model, X_test, metrics):
        """Calculate all metrics for a single fold."""
        metrics['accuracy'].append(accuracy_score(y_true, y_pred))
        metrics['precision_macro'].append(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall_macro'].append(recall_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1_macro'].append(f1_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['precision_weighted'].append(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['recall_weighted'].append(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['f1_weighted'].append(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        # ROC-AUC (handle models without predict_proba)
        try:
            y_proba = model.predict_proba(X_test)
            metrics['roc_auc_ovr'].append(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        except (AttributeError, ValueError):
            metrics['roc_auc_ovr'].append(np.nan)
    
    def _aggregate_metrics(self, fold_metrics: Dict[str, List]) -> Dict[str, float]:
        """Aggregate fold metrics into mean and standard deviation."""
        results = {}
        for metric, values in fold_metrics.items():
            results[f'{metric}_mean'] = np.mean(values)
            results[f'{metric}_std'] = np.std(values)
        return results

class BenchmarkRunner:
    """Main benchmark execution class."""
    
    def __init__(self, models: Dict, evaluator: ModelEvaluator, 
                 results_path: str, checkpoint_frequency: int = 1):
        self.models = models
        self.evaluator = evaluator
        self.results_path = results_path
        self.checkpoint_frequency = checkpoint_frequency
    
    def run_benchmark(self, X: np.ndarray, y: np.ndarray, 
                     resume: bool = True) -> pd.DataFrame:
        """
        Execute comprehensive benchmark with checkpointing.
        
        Args:
            X: Feature matrix
            y: Target vector
            resume: Whether to resume from existing results
            
        Returns:
            DataFrame with benchmark results
        """
        # Load existing results if resuming
        if resume and Path(self.results_path).exists():
            results_df = pd.read_csv(self.results_path)
            completed_models = set(results_df['Model'].tolist())
        else:
            results_df = pd.DataFrame()
            completed_models = set()
        
        new_results = []
        
        for i, (model_name, model) in enumerate(self.models.items()):
            if model_name in completed_models:
                logger.info(f"Skipping {model_name} (already completed)")
                continue
            
            logger.info(f"Running benchmark for: {model_name}")
            
            try:
                metrics = self.evaluator.evaluate_model(model, X, y)
                metrics['Model'] = model_name
                new_results.append(metrics)
                
                # Checkpoint saving
                if (i + 1) % self.checkpoint_frequency == 0:
                    self._save_checkpoint(results_df, new_results)
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        # Final save
        final_results = pd.concat([results_df, pd.DataFrame(new_results)], 
                                 ignore_index=True)
        final_results.to_csv(self.results_path, index=False)
        
        return final_results
    
    def _save_checkpoint(self, existing_df: pd.DataFrame, new_results: List[Dict]):
        """Save checkpoint of current results."""
        current_results = pd.concat([existing_df, pd.DataFrame(new_results)], 
                                   ignore_index=True)
        current_results.to_csv(self.results_path, index=False)
        logger.info(f"Checkpoint saved to {self.results_path}")
