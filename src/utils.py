"""
Utility functions for the BEED benchmark pipeline.
"""

import logging
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

def setup_logging(level: int = logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('benchmark.log'),
            logging.StreamHandler()
        ]
    )

def save_model(model: Any, filepath: Path, metadata: Dict[str, Any] = None):
    """Save model with metadata."""
    joblib.dump(model, filepath)
    
    if metadata:
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

def load_model(filepath: Path) -> tuple:
    """Load model and metadata."""
    model = joblib.load(filepath)
    
    metadata_path = filepath.with_suffix('.json')
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata

def create_results_summary(results_df: pd.DataFrame, 
                          output_path: Path) -> Dict[str, Any]:
    """Create comprehensive results summary."""
    summary = {
        'best_model': results_df.loc[results_df['f1_weighted_mean'].idxmax(), 'Model'],
        'performance_summary': results_df[['Model', 'f1_weighted_mean', 'f1_weighted_std']].to_dict('records'),
        'ranking': results_df.sort_values('f1_weighted_mean', ascending=False)['Model'].tolist()
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def plot_benchmark_results(results_df: pd.DataFrame, 
                          output_path: Path, 
                          figsize: tuple = (12, 8)):
    """Create publication-ready benchmark results plot."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Sort by performance
    results_sorted = results_df.sort_values('f1_weighted_mean', ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot with error bars
    bars = ax.barh(
        results_sorted['Model'], 
        results_sorted['f1_weighted_mean'],
        xerr=results_sorted['f1_weighted_std'], 
        capsize=5, 
        color=sns.color_palette('viridis', len(results_sorted))
    )
    
    # Formatting
    ax.set_xlabel('Weighted F1-Score', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title('Model Performance Comparison (10-Fold CV)', fontsize=14)
    ax.set_xlim(0, 1.05)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
