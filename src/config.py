"""
Configuration module for BEED benchmark pipeline.
Contains all project settings, paths, and model parameters.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class ProjectPaths:
    """Project directory structure configuration."""
    BASE_PATH: Path = Path("/content/drive/My Drive/Colab_Projects/BEED_Benchmark/")
    DATA_DIR: Path = BASE_PATH / "data" / "raw"
    PROCESSED_DIR: Path = BASE_PATH / "data" / "processed"
    RESULTS_DIR: Path = BASE_PATH / "results"
    MODELS_DIR: Path = BASE_PATH / "results" / "models"
    FIGURES_DIR: Path = BASE_PATH / "results" / "figures"
    
    def create_directories(self):
        """Create all necessary directories if they don't exist."""
        for path in [self.DATA_DIR, self.PROCESSED_DIR, self.RESULTS_DIR, 
                     self.MODELS_DIR, self.FIGURES_DIR]:
            path.mkdir(parents=True, exist_ok=True)

@dataclass
class ExperimentConfig:
    """Experiment configuration parameters."""
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 10
    TEST_SIZE: float = 0.2
    N_ITER_RANDOM_SEARCH: int = 50
    MAX_SHAP_SAMPLES: int = 2000
    FEATURE_SELECTION_CV_FOLDS: int = 5
    FEATURE_SELECTION_RUNS: int = 3

# Global configuration instances
PATHS = ProjectPaths()
CONFIG = ExperimentConfig()
