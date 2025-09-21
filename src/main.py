"""
Main execution script for BEED benchmark pipeline.
"""

import logging
from pathlib import Path
import argparse

from config import PATHS, CONFIG
from data_preprocessing import DataPreprocessor
from model_definitions import ModelFactory
from evaluation import ModelEvaluator, BenchmarkRunner
from feature_selection import FeatureSelector
from hyperparameter_tuning import HyperparameterTuner
from utils import setup_logging, save_results

def main():
    """Main execution function."""
    # Setup
    setup_logging()
    PATHS.create_directories()
    
    # Data preprocessing
    preprocessor = DataPreprocessor(
        PATHS.DATA_DIR / "BEED_Data.csv",
        PATHS.PROCESSED_DIR / "preprocessed_data.csv"
    )
    X, y = preprocessor.preprocess_data()
    
    # Model definitions
    models = ModelFactory.get_base_models()
    
    # Benchmark evaluation
    evaluator = ModelEvaluator(cv_folds=CONFIG.CV_FOLDS, 
                              random_state=CONFIG.RANDOM_STATE)
    
    benchmark_runner = BenchmarkRunner(
        models=models,
        evaluator=evaluator,
        results_path=PATHS.RESULTS_DIR / "benchmark_results.csv"
    )
    
    results = benchmark_runner.run_benchmark(X, y)
    
    # Feature selection (optional)
    if args.feature_selection:
        feature_selector = FeatureSelector(
            estimator_name="XGBoost",
            cv_folds=CONFIG.FEATURE_SELECTION_CV_FOLDS,
            n_runs=CONFIG.FEATURE_SELECTION_RUNS
        )
        X_selected = feature_selector.select_features(X, y)
    
    # Hyperparameter tuning (optional)
    if args.hyperparameter_tuning:
        tuner = HyperparameterTuner(
            param_grids=ModelFactory.get_hyperparameter_grids(),
            cv_folds=5,
            n_iter=CONFIG.N_ITER_RANDOM_SEARCH
        )
        tuning_results = tuner.tune_models(X, y)
    
    logging.info("Benchmark completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEED Benchmark Pipeline")
    parser.add_argument("--feature-selection", action="store_true", 
                       help="Perform feature selection")
    parser.add_argument("--hyperparameter-tuning", action="store_true", 
                       help="Perform hyperparameter tuning")
    
    args = parser.parse_args()
    main()
