# BEED Benchmark: A Modular Machine Learning Pipeline for EEG-based Epileptic Seizure Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **BEED** (Biomedical EEG Epilepsy Detection) Benchmark is a comprehensive, modular implementation for benchmarking machine learning algorithms on EEG-based epileptic seizure detection datasets. This pipeline provides standardized evaluation protocols, reproducible results, and extensible architecture for epilepsy research.

## 🚀 Key Features

- **🔬 Comprehensive Evaluation**: 10-fold stratified cross-validation with multiple metrics
- **🧠 Multiple ML Algorithms**: Support for 9+ state-of-the-art machine learning models
- **⚡ Hyperparameter Optimization**: Automated hyperparameter tuning with random search
- **🎯 Feature Selection**: Advanced feature selection techniques with stability analysis
- **📊 Rich Visualizations**: Publication-ready plots and comprehensive result analysis
- **🔄 Reproducible**: Fixed random seeds and version-controlled dependencies
- **💾 Checkpointing**: Resume long-running experiments from interruptions
- **📈 External Validation**: Cross-dataset validation capabilities
- **🐍 Modern Python**: Type hints, dataclasses, and clean architecture

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Supported Models](#supported-models)
- [Results & Evaluation](#results--evaluation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for XGBoost acceleration

### Install from PyPI (Recommended)

```bash
pip install beed-benchmark
```

### Install from Source

```bash
git clone https://github.com/engkinandatama/beed-benchmark.git
cd beed-benchmark
pip install -r requirements.txt
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/engkinandatama/beed-benchmark.git
cd beed-benchmark
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

## 🚀 Quick Start

### Basic Usage

```python
from beed_benchmark import BEEDBenchmark
from beed_benchmark.config import PATHS, CONFIG

# Initialize benchmark
benchmark = BEEDBenchmark(
    data_path="path/to/your/eeg_data.csv",
    output_dir="results/"
)

# Run complete benchmark
results = benchmark.run_full_pipeline()

# View results
print(f"Best model: {results.best_model}")
print(f"Best F1-score: {results.best_score:.4f}")
```

### Command Line Interface

```bash
# Run basic benchmark
python -m beed_benchmark --data-path data/BEED_Data.csv --output-dir results/

# Run with hyperparameter tuning
python -m beed_benchmark --data-path data/BEED_Data.csv --tune-hyperparameters

# Run with feature selection
python -m beed_benchmark --data-path data/BEED_Data.csv --feature-selection

# Resume interrupted experiment
python -m beed_benchmark --data-path data/BEED_Data.csv --resume
```

### Jupyter Notebook

```python
# Load example notebook
from beed_benchmark.examples import load_example_notebook
load_example_notebook("01_basic_benchmark.ipynb")
```

## 📁 Project Structure

```
beed_benchmark/
├── 📂 src/beed_benchmark/          # Main package
│   ├── __init__.py                 # Package initialization
│   ├── config.py                   # Configuration management
│   ├── data_preprocessing.py       # Data loading & preprocessing
│   ├── model_definitions.py        # ML model definitions
│   ├── feature_selection.py        # Feature selection algorithms
│   ├── hyperparameter_tuning.py    # Hyperparameter optimization
│   ├── evaluation.py               # Model evaluation & metrics
│   ├── external_validation.py      # Cross-dataset validation
│   ├── visualization.py            # Plotting & visualization
│   └── utils.py                    # Utility functions
├── 📂 notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb   # Exploratory data analysis
│   ├── 02_benchmark_execution.ipynb # Main benchmark execution
│   ├── 03_results_analysis.ipynb   # Results analysis
│   └── 04_external_validation.ipynb # External validation
├── 📂 data/                        # Data directory
│   ├── raw/                        # Raw datasets
│   ├── processed/                  # Processed datasets
│   └── external/                   # External validation datasets
├── 📂 results/                     # Results directory
│   ├── models/                     # Trained models
│   ├── metrics/                    # Evaluation metrics
│   ├── figures/                    # Generated plots
│   └── reports/                    # Analysis reports
├── 📂 tests/                       # Unit tests
├── 📂 docs/                        # Documentation
├── 📂 examples/                    # Example scripts
├── requirements.txt                # Dependencies
├── requirements-dev.txt            # Development dependencies
├── setup.py                        # Package setup
├── pyproject.toml                  # Project configuration
├── .github/                        # GitHub workflows
└── README.md                       # This file
```

## 📖 Usage Guide

### 1. Data Preparation

Your EEG dataset should be in CSV format with features in columns and the target variable in the last column:

```csv
feature_1,feature_2,...,feature_n,target
0.123,0.456,...,0.789,0
0.234,0.567,...,0.890,1
...
```

Classes should be encoded as:
- `0`: Non-seizure (normal/interictal)
- `1`: Seizure (ictal)
- `2`: Pre-seizure (preictal) - if applicable

### 2. Configuration

Customize your experiment settings:

```python
from beed_benchmark.config import CONFIG

# Experiment settings
CONFIG.CV_FOLDS = 10              # Cross-validation folds
CONFIG.RANDOM_STATE = 42          # Reproducibility seed
CONFIG.N_ITER_RANDOM_SEARCH = 100 # Hyperparameter search iterations

# Performance settings
CONFIG.N_JOBS = -1                # Parallel processing
CONFIG.MEMORY_LIMIT = "8GB"       # Memory limit per job
```

### 3. Model Selection

Choose specific models to benchmark:

```python
from beed_benchmark import BEEDBenchmark

benchmark = BEEDBenchmark(
    models=['Random Forest', 'XGBoost', 'SVM', 'Neural Network']
)
```

### 4. Advanced Features

#### Feature Selection

```python
from beed_benchmark.feature_selection import FeatureSelector

selector = FeatureSelector(
    method='rfe',           # Recursive Feature Elimination
    estimator='xgboost',    # Base estimator
    n_features=100          # Number of features to select
)

X_selected = selector.fit_transform(X, y)
```

#### Hyperparameter Tuning

```python
from beed_benchmark.hyperparameter_tuning import HyperparameterTuner

tuner = HyperparameterTuner(
    search_method='random',  # 'random', 'grid', 'bayesian'
    cv_folds=5,
    n_iter=50
)

best_params = tuner.optimize(model, X, y)
```

#### External Validation

```python
from beed_benchmark.external_validation import ExternalValidator

validator = ExternalValidator()
external_results = validator.validate(
    trained_models,
    external_dataset_path="data/external/test_data.csv"
)
```

## ⚙️ Configuration

The pipeline uses a hierarchical configuration system:

### Project Paths (`ProjectPaths`)
```python
BASE_PATH: Path          # Project root directory
DATA_DIR: Path          # Raw data location
PROCESSED_DIR: Path     # Processed data cache
RESULTS_DIR: Path       # Results output
MODELS_DIR: Path        # Saved models
FIGURES_DIR: Path       # Generated plots
```

### Experiment Settings (`ExperimentConfig`)
```python
RANDOM_STATE: int = 42                    # Reproducibility seed
CV_FOLDS: int = 10                       # Cross-validation folds
TEST_SIZE: float = 0.2                   # Train/test split ratio
N_ITER_RANDOM_SEARCH: int = 50          # Hyperparameter search iterations
MAX_SHAP_SAMPLES: int = 2000            # SHAP analysis sample limit
FEATURE_SELECTION_CV_FOLDS: int = 5     # Feature selection CV folds
FEATURE_SELECTION_RUNS: int = 3         # Feature selection stability runs
```

## 🤖 Supported Models

| Model | Type | Hyperparameter Tuning | Feature Importance | Probability Output |
|-------|------|----------------------|-------------------|-------------------|
| **Logistic Regression** | Linear | ✅ | ✅ | ✅ |
| **K-Nearest Neighbors** | Instance-based | ✅ | ❌ | ✅ |
| **Support Vector Machine** | Kernel-based | ✅ | ❌ | ✅ |
| **Decision Tree** | Tree-based | ✅ | ✅ | ✅ |
| **Random Forest** | Ensemble | ✅ | ✅ | ✅ |
| **Gradient Boosting** | Ensemble | ✅ | ✅ | ✅ |
| **XGBoost** | Ensemble | ✅ | ✅ | ✅ |
| **LightGBM** | Ensemble | ✅ | ✅ | ✅ |
| **Multi-layer Perceptron** | Neural Network | ✅ | ❌ | ✅ |

### Adding Custom Models

```python
from beed_benchmark.model_definitions import ModelFactory

# Define custom model
class CustomModelFactory(ModelFactory):
    @staticmethod
    def get_custom_model():
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(random_state=42)
    
    @staticmethod
    def get_custom_hyperparameters():
        return {
            'n_estimators': randint(100, 500),
            'max_depth': [10, 20, None],
            'min_samples_split': randint(2, 11)
        }

# Register custom model
ModelFactory.register_model('Extra Trees', CustomModelFactory.get_custom_model())
```

## 📊 Results & Evaluation

### Evaluation Metrics

The benchmark evaluates models using comprehensive metrics:

**Classification Metrics:**
- Accuracy
- Precision (macro & weighted)
- Recall (macro & weighted) 
- F1-score (macro & weighted)
- ROC-AUC (one-vs-rest)

**Performance Metrics:**
- Training time
- Inference time
- Memory usage

**Statistical Analysis:**
- Mean ± Standard deviation across CV folds
- Statistical significance testing
- Effect size analysis

### Results Format

```python
# Results DataFrame structure
results = pd.DataFrame({
    'Model': str,                    # Model name
    'accuracy_mean': float,          # Mean accuracy
    'accuracy_std': float,           # Accuracy standard deviation
    'f1_weighted_mean': float,       # Mean weighted F1-score
    'f1_weighted_std': float,        # F1-score standard deviation
    'train_time_mean': float,        # Mean training time (seconds)
    'inference_time_mean': float,    # Mean inference time (seconds)
    # ... additional metrics
})
```

### Visualization

The pipeline generates publication-ready visualizations:

- **Performance Comparison**: Horizontal bar charts with error bars
- **ROC Curves**: Multi-class ROC analysis
- **Confusion Matrices**: Detailed classification results
- **Feature Importance**: Top contributing features
- **Learning Curves**: Model convergence analysis
- **Statistical Tests**: Significance testing results

## 🔬 Example Results

Based on evaluation of EEG epilepsy detection datasets:

| Model | Weighted F1-Score | Accuracy | Training Time (s) | Inference Time (ms) |
|-------|------------------|----------|------------------|-------------------|
| **XGBoost** | 0.924 ± 0.012 | 0.925 ± 0.011 | 2.34 ± 0.23 | 1.2 ± 0.1 |
| **Random Forest** | 0.918 ± 0.015 | 0.919 ± 0.014 | 1.89 ± 0.15 | 0.8 ± 0.1 |
| **LightGBM** | 0.916 ± 0.013 | 0.917 ± 0.012 | 1.45 ± 0.12 | 0.9 ± 0.1 |
| **SVM** | 0.908 ± 0.018 | 0.909 ± 0.017 | 12.4 ± 1.2 | 4.5 ± 0.3 |
| **Neural Network** | 0.895 ± 0.021 | 0.896 ± 0.020 | 8.7 ± 0.8 | 2.1 ± 0.2 |

> Results may vary based on dataset characteristics and preprocessing methods.

## 📚 Documentation

Comprehensive documentation is available:

- **[API Reference](./docs/api/)**: Complete API documentation
- **[User Guide](./docs/user_guide/)**: Step-by-step tutorials
- **[Examples](./examples/)**: Code examples and use cases
- **[Paper Supplements](./docs/paper/)**: Reproducible research materials

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=beed_benchmark

# Run specific test category
pytest tests/test_models.py
pytest tests/test_evaluation.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Citation

If you use BEED Benchmark in your research, please cite our work:

```bibtex
@software{beed_benchmark2024,
  title={BEED Benchmark: A Modular Machine Learning Pipeline for EEG-based Epileptic Seizure Detection},
  author={Nandatama, Engki},
  year={2025},
  publisher={GitHub},
  url={https://github.com/engkinandatama/beed-benchmark},
  version={1.0.0}
}
```
<!--
### Related Publications

```bibtex
@article{your_paper2024,
  title={A Comprehensive Benchmark and Robustness Analysis of Machine Learning Models for Epileptic Seizure Classification on the BEED Dataset},
  author={Nandatama, Engki},
  journal={Journal Name},
  year={2025},
  doi={10.xxxx/xxxx}
}
```
-->
## 🙏 Acknowledgments

- **Contributors**: Thank you to all contributors who have helped improve this project
- **Data Providers**: Thanks to institutions providing EEG datasets for research
- **Open Source Community**: Built on excellent open-source libraries including scikit-learn, XGBoost, and pandas
- **Research Community**: Inspired by best practices from machine learning and epilepsy research communities

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/engkinandatama/beed-benchmark/issues)
- **Discussions**: [GitHub Discussions](https://github.com/engkinandatama/beed-benchmark/discussions)

## 🔄 Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and updates.

---

<div align="center">

[⭐ Star this project](https://github.com/engkinandatama/beed-benchmark) if you find it useful!

</div>
