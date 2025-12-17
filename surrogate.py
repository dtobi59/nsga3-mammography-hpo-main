"""
Gaussian Process Surrogate Model for Multi-Objective Optimization

Author: David
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler


class MultiObjectiveGPSurrogate:
    """
    Independent Gaussian Process surrogate for each objective.
    Uses Matern 5/2 kernel (better for non-smooth DL landscapes).
    """
    
    def __init__(self, n_objectives: int = 5, kernel_type: str = 'matern'):
        self.n_objectives = n_objectives
        self.kernel_type = kernel_type
        
        self.models = []
        self.scalers_X = []
        self.scalers_y = []
        self.is_fitted = False
        
        # Initialize GPs
        for _ in range(n_objectives):
            if kernel_type == 'matern':
                kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            else:
                kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5
            )
            self.models.append(gp)
            self.scalers_X.append(StandardScaler())
            self.scalers_y.append(StandardScaler())
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit GP models to data.
        
        Args:
            X: (n_samples, n_features) decision variables
            y: (n_samples, n_objectives) objective values
        """
        for i in range(self.n_objectives):
            X_scaled = self.scalers_X[i].fit_transform(X)
            y_scaled = self.scalers_y[i].fit_transform(y[:, i:i+1]).ravel()
            self.models[i].fit(X_scaled, y_scaled)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict objectives with uncertainty.
        
        Returns:
            means: (n_samples, n_objectives)
            stds: (n_samples, n_objectives)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        n_samples = X.shape[0]
        means = np.zeros((n_samples, self.n_objectives))
        stds = np.zeros((n_samples, self.n_objectives))
        
        for i in range(self.n_objectives):
            X_scaled = self.scalers_X[i].transform(X)
            mean_scaled, std_scaled = self.models[i].predict(X_scaled, return_std=True)
            
            # Inverse transform mean
            means[:, i] = self.scalers_y[i].inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
            # Scale std appropriately
            stds[:, i] = std_scaled * self.scalers_y[i].scale_[0]
        
        return means, stds


class SurrogateAssistedSelection:
    """
    Manages surrogate-assisted evaluation selection.
    
    Strategy: Select top candidates by acquisition value (predicted + uncertainty)
    for true evaluation, use surrogate for rest.
    """
    
    def __init__(
        self,
        surrogate: MultiObjectiveGPSurrogate,
        true_eval_ratio: float = 0.3,
        min_samples: int = 15,
        objectives_maximize: List[bool] = None
    ):
        """
        Args:
            surrogate: GP surrogate model
            true_eval_ratio: Fraction of population to truly evaluate
            min_samples: Minimum true evaluations before using surrogate
            objectives_maximize: Which objectives to maximize (vs minimize)
        """
        self.surrogate = surrogate
        self.true_eval_ratio = true_eval_ratio
        self.min_samples = min_samples
        self.objectives_maximize = objectives_maximize or [True, True, True, False, False]
        
        self.X_evaluated = None
        self.y_evaluated = None
        self.n_true_evals = 0
    
    def should_use_surrogate(self) -> bool:
        """Check if we have enough data to use surrogate."""
        return self.n_true_evals >= self.min_samples
    
    def register_true_evaluation(self, X: np.ndarray, y: np.ndarray):
        """Register results of true evaluation."""
        if self.X_evaluated is None:
            self.X_evaluated = X.copy()
            self.y_evaluated = y.copy()
        else:
            self.X_evaluated = np.vstack([self.X_evaluated, X])
            self.y_evaluated = np.vstack([self.y_evaluated, y])
        
        self.n_true_evals += X.shape[0]
    
    def update_surrogate(self):
        """Update surrogate model with collected data."""
        if self.X_evaluated is not None and len(self.X_evaluated) >= self.min_samples:
            self.surrogate.fit(self.X_evaluated, self.y_evaluated)
    
    def select_for_true_evaluation(self, X: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """
        Select which candidates to truly evaluate.
        
        Returns:
            indices: List of indices to evaluate
            X_selected: The selected candidates
        """
        n_samples = X.shape[0]
        n_true = max(1, int(n_samples * self.true_eval_ratio))
        
        if not self.should_use_surrogate():
            # Evaluate all
            return list(range(n_samples)), X
        
        # Get predictions and uncertainty
        means, stds = self.surrogate.predict(X)
        
        # Compute acquisition value (UCB-style)
        # Higher is better for selection
        acquisition = np.zeros(n_samples)
        
        for i in range(self.surrogate.n_objectives):
            if self.objectives_maximize[i]:
                acquisition += means[:, i] + 0.1 * stds[:, i]
            else:
                acquisition += -means[:, i] + 0.1 * stds[:, i]
        
        # Also add diversity bonus (uncertainty)
        acquisition += 0.5 * stds.mean(axis=1)
        
        # Select top candidates
        indices = np.argsort(acquisition)[-n_true:].tolist()
        
        return indices, X[indices]
    
    def get_combined_fitness(
        self,
        X: np.ndarray,
        true_indices: List[int],
        true_fitness: np.ndarray
    ) -> np.ndarray:
        """
        Combine true evaluations with surrogate predictions.
        
        Returns:
            fitness: (n_samples, n_objectives)
        """
        n_samples = X.shape[0]
        fitness = np.zeros((n_samples, self.surrogate.n_objectives))
        
        # Get surrogate predictions for all
        if self.surrogate.is_fitted:
            means, _ = self.surrogate.predict(X)
            fitness = means.copy()
        
        # Override with true evaluations
        for i, idx in enumerate(true_indices):
            fitness[idx] = true_fitness[i]
        
        return fitness
