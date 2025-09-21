"""
Machine learning model definitions and configurations.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import uniform, randint, loguniform
from typing import Dict, Any
import warnings

warnings.filterwarnings('ignore')

class ModelFactory:
    """Factory class for creating and managing ML models."""
    
    @staticmethod
    def get_base_models() -> Dict[str, Any]:
        """Returns dictionary of base models with default parameters."""
        return {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "SVM": Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(random_state=42, probability=True))
            ]),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, 
                                   eval_metric='logloss'),
            "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
            "MLP": Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPClassifier(random_state=42, max_iter=500))
            ])
        }
    
    @staticmethod
    def get_hyperparameter_grids() -> Dict[str, Dict[str, Any]]:
        """Returns hyperparameter search spaces for each model."""
        return {
            "Random Forest": {
                'n_estimators': randint(100, 500),
                'max_depth': [10, 20, 30, 40, None],
                'min_samples_split': randint(2, 11),
                'min_samples_leaf': randint(1, 5),
                'max_features': ['sqrt', 'log2']
            },
            "XGBoost": {
                'n_estimators': randint(100, 500),
                'learning_rate': uniform(0.01, 0.2),
                'max_depth': randint(3, 10),
                'subsample': uniform(0.7, 0.3),
                'colsample_bytree': uniform(0.7, 0.3)
            },
            "LightGBM": {
                'n_estimators': randint(100, 500),
                'learning_rate': uniform(0.01, 0.2),
                'num_leaves': randint(20, 50),
                'max_depth': [-1, 10, 20, 30],
                'subsample': uniform(0.7, 0.3),
                'colsample_bytree': uniform(0.7, 0.3)
            },
            "SVM": {
                'svc__C': loguniform(1e-1, 1e3),
                'svc__gamma': loguniform(1e-4, 1e-1),
                'svc__kernel': ['rbf']
            },
            "MLP": {
                'mlp__hidden_layer_sizes': [(50,50), (100,), (100, 50, 25)],
                'mlp__activation': ['relu', 'tanh'],
                'mlp__alpha': loguniform(1e-5, 1e-2),
                'mlp__learning_rate': ['constant', 'adaptive']
            }
        }
