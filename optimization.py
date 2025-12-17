"""
NSGA-III Multi-Objective Optimization for CNN Hyperparameters

Author: David
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional
import pickle
import os
from pathlib import Path

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from config import HyperparameterSpace, NSGAIIIConfig


class HyperparameterEncoder:
    """Encodes/decodes hyperparameters to/from continuous representation."""
    
    def __init__(self, hp_space: HyperparameterSpace):
        self.hp_space = hp_space
        self.var_info = self._build_variables()
        self.n_vars = len(self.var_info)
    
    def _build_variables(self) -> List[Dict]:
        """Define all variables with their types and bounds."""
        variables = [
            # Categorical (encoded as integers)
            {'name': 'backbone', 'type': 'cat', 'options': self.hp_space.backbone},
            {'name': 'unfreeze_strategy', 'type': 'cat', 'options': self.hp_space.unfreeze_strategy},
            {'name': 'fc_hidden_size', 'type': 'cat', 'options': self.hp_space.fc_hidden_size},
            {'name': 'use_additional_fc', 'type': 'bool'},
            {'name': 'optimizer', 'type': 'cat', 'options': self.hp_space.optimizer},
            {'name': 'scheduler', 'type': 'cat', 'options': self.hp_space.scheduler},
            {'name': 'batch_size', 'type': 'cat', 'options': self.hp_space.batch_size},
            {'name': 'loss_function', 'type': 'cat', 'options': self.hp_space.loss_function},
            {'name': 'class_weight_strategy', 'type': 'cat', 'options': self.hp_space.class_weight_strategy},
            {'name': 'horizontal_flip', 'type': 'bool'},
            {'name': 'use_mixup', 'type': 'bool'},
            # Continuous
            {'name': 'dropout_rate', 'type': 'float', 'bounds': self.hp_space.dropout_rate},
            {'name': 'learning_rate', 'type': 'float_log', 'bounds': self.hp_space.learning_rate},
            {'name': 'weight_decay', 'type': 'float_log', 'bounds': self.hp_space.weight_decay},
            {'name': 'focal_gamma', 'type': 'float', 'bounds': self.hp_space.focal_gamma},
            {'name': 'oversampling_ratio', 'type': 'float', 'bounds': self.hp_space.oversampling_ratio},
            {'name': 'brightness_contrast', 'type': 'float', 'bounds': self.hp_space.brightness_contrast},
            {'name': 'mixup_alpha', 'type': 'float', 'bounds': self.hp_space.mixup_alpha},
            # Integer
            {'name': 'epochs', 'type': 'int', 'bounds': self.hp_space.epochs},
            {'name': 'rotation_range', 'type': 'int', 'bounds': self.hp_space.rotation_range},
        ]
        return variables
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get lower and upper bounds for pymoo."""
        xl, xu = [], []
        for v in self.var_info:
            if v['type'] == 'cat':
                xl.append(0)
                xu.append(len(v['options']) - 1)
            elif v['type'] == 'bool':
                xl.append(0)
                xu.append(1)
            elif v['type'] in ['float', 'float_log']:
                xl.append(v['bounds'][0])
                xu.append(v['bounds'][1])
            elif v['type'] == 'int':
                xl.append(v['bounds'][0])
                xu.append(v['bounds'][1])
        return np.array(xl), np.array(xu)
    
    def decode(self, x: np.ndarray) -> Dict[str, Any]:
        """Decode continuous array to config dict."""
        config = {}
        
        for i, v in enumerate(self.var_info):
            val = x[i]
            
            if v['type'] == 'cat':
                idx = int(np.clip(np.round(val), 0, len(v['options']) - 1))
                config[v['name']] = v['options'][idx]
            elif v['type'] == 'bool':
                config[v['name']] = bool(np.round(val))
            elif v['type'] == 'float':
                config[v['name']] = float(np.clip(val, v['bounds'][0], v['bounds'][1]))
            elif v['type'] == 'float_log':
                config[v['name']] = float(np.clip(val, v['bounds'][0], v['bounds'][1]))
            elif v['type'] == 'int':
                config[v['name']] = int(np.clip(np.round(val), v['bounds'][0], v['bounds'][1]))
        
        return config


class MammographyHPOProblem(Problem):
    """
    Multi-objective optimization problem for mammography CNN hyperparameters.
    
    Objectives: sensitivity, specificity, AUC (maximize), model_size, inference_time (minimize)
    """
    
    def __init__(
        self,
        hp_space: HyperparameterSpace,
        eval_function: Callable,
        surrogate_selector = None,
        n_objectives: int = 5
    ):
        self.encoder = HyperparameterEncoder(hp_space)
        self.eval_function = eval_function
        self.surrogate_selector = surrogate_selector
        
        xl, xu = self.encoder.get_bounds()
        
        super().__init__(
            n_var=self.encoder.n_vars,
            n_obj=n_objectives,
            n_constr=0,
            xl=xl,
            xu=xu
        )
        
        self.eval_count = 0
        self.true_eval_count = 0
        self.surrogate_eval_count = 0
        
        # Objectives to maximize (will negate for pymoo minimization)
        self.maximize_mask = np.array([True, True, True, False, False])
    
    def _evaluate(self, X: np.ndarray, out: Dict, *args, **kwargs):
        """Evaluate population."""
        n_samples = X.shape[0]
        F = np.zeros((n_samples, self.n_obj))
        
        use_surrogate = (
            self.surrogate_selector is not None and 
            self.surrogate_selector.should_use_surrogate()
        )
        
        if use_surrogate:
            # Select subset for true evaluation
            true_indices, _ = self.surrogate_selector.select_for_true_evaluation(X)
            
            # True evaluation
            true_fitness = np.zeros((len(true_indices), self.n_obj))
            for i, idx in enumerate(true_indices):
                config = self.encoder.decode(X[idx])
                results = self.eval_function(config)
                true_fitness[i] = self._results_to_array(results)
                
                self.surrogate_selector.register_true_evaluation(
                    X[idx:idx+1], true_fitness[i:i+1]
                )
            
            self.true_eval_count += len(true_indices)
            self.surrogate_selector.update_surrogate()
            
            # Combined fitness
            F = self.surrogate_selector.get_combined_fitness(X, true_indices, true_fitness)
            self.surrogate_eval_count += n_samples - len(true_indices)
        else:
            # Full true evaluation
            for i in range(n_samples):
                config = self.encoder.decode(X[i])
                results = self.eval_function(config)
                F[i] = self._results_to_array(results)
                
                if self.surrogate_selector is not None:
                    self.surrogate_selector.register_true_evaluation(X[i:i+1], F[i:i+1])
            
            self.true_eval_count += n_samples
            
            if self.surrogate_selector is not None:
                self.surrogate_selector.update_surrogate()
        
        self.eval_count += n_samples
        
        # Convert to minimization
        F_min = F.copy()
        F_min[:, self.maximize_mask] = -F_min[:, self.maximize_mask]
        
        out["F"] = F_min
    
    def _results_to_array(self, results: Dict) -> np.ndarray:
        """Convert results dict to array."""
        return np.array([
            results['sensitivity'],
            results['specificity'],
            results['auc'],
            results['model_size'],
            results['inference_time']
        ])


class OptimizationCallback(Callback):
    """Callback for logging and checkpointing."""
    
    def __init__(self, checkpoint_dir: str, checkpoint_freq: int = 5, verbose: bool = True):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_freq = checkpoint_freq
        self.verbose = verbose
        self.history = {'generation': [], 'best_auc': [], 'pareto_size': []}
    
    def notify(self, algorithm):
        """Called after each generation."""
        gen = algorithm.n_gen
        problem = algorithm.problem
        F = algorithm.pop.get("F")
        
        # Convert back from minimization
        F_orig = F.copy()
        F_orig[:, problem.maximize_mask] = -F_orig[:, problem.maximize_mask]
        
        self.history['generation'].append(gen)
        self.history['best_auc'].append(F_orig[:, 2].max())
        self.history['pareto_size'].append(len(F))
        
        if self.verbose:
            print(f"\nGen {gen}: Evals={problem.eval_count} (True={problem.true_eval_count}), "
                  f"Best AUC={F_orig[:, 2].max():.4f}, Pareto size={len(F)}")
        
        if gen % self.checkpoint_freq == 0:
            self._save_checkpoint(algorithm, gen)
    
    def _save_checkpoint(self, algorithm, gen: int):
        """Save checkpoint."""
        path = self.checkpoint_dir / f"checkpoint_gen_{gen}.pkl"
        with open(path, 'wb') as f:
            pickle.dump({
                'generation': gen,
                'history': self.history,
                'eval_count': algorithm.problem.eval_count
            }, f)
        if self.verbose:
            print(f"  Checkpoint saved: {path}")


def run_optimization(
    hp_space: HyperparameterSpace,
    nsga_config: NSGAIIIConfig,
    eval_function: Callable,
    output_dir: str,
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run NSGA-III optimization.
    
    Args:
        hp_space: Search space
        nsga_config: Algorithm config
        eval_function: Function that takes config dict, returns dict with objectives
        output_dir: Where to save results
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Dict with pareto_X, pareto_F, pareto_configs, history
    """
    from surrogate import MultiObjectiveGPSurrogate, SurrogateAssistedSelection
    
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create encoder
    encoder = HyperparameterEncoder(hp_space)
    
    # Create surrogate
    surrogate = MultiObjectiveGPSurrogate(n_objectives=nsga_config.n_objectives)
    selector = SurrogateAssistedSelection(
        surrogate=surrogate,
        true_eval_ratio=1.0 - nsga_config.surrogate_ratio,
        min_samples=nsga_config.min_samples_for_surrogate,
        objectives_maximize=[True, True, True, False, False]
    )
    
    # Create problem
    problem = MammographyHPOProblem(
        hp_space=hp_space,
        eval_function=eval_function,
        surrogate_selector=selector,
        n_objectives=nsga_config.n_objectives
    )
    
    # Create NSGA-III
    ref_dirs = get_reference_directions("das-dennis", nsga_config.n_objectives, n_partitions=nsga_config.n_partitions)
    
    algorithm = NSGA3(
        ref_dirs=ref_dirs,
        pop_size=nsga_config.pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=nsga_config.crossover_prob, eta=nsga_config.crossover_eta),
        mutation=PM(prob=1.0/encoder.n_vars, eta=nsga_config.mutation_eta),
        eliminate_duplicates=True
    )
    
    # Callback
    callback = OptimizationCallback(
        checkpoint_dir=os.path.join(output_dir, 'checkpoints'),
        checkpoint_freq=nsga_config.checkpoint_frequency,
        verbose=verbose
    )
    
    # Run
    if verbose:
        print(f"Starting NSGA-III: pop_size={nsga_config.pop_size}, generations={nsga_config.n_generations}")
        print(f"Reference directions: {len(ref_dirs)}")
    
    result = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", nsga_config.n_generations),
        seed=seed,
        callback=callback,
        verbose=False
    )
    
    # Process results
    X_pareto = result.X
    F_pareto = result.F.copy()
    F_pareto[:, problem.maximize_mask] = -F_pareto[:, problem.maximize_mask]
    
    pareto_configs = [encoder.decode(x) for x in X_pareto]
    
    # Save
    final_results = {
        'pareto_X': X_pareto,
        'pareto_F': F_pareto,
        'pareto_configs': pareto_configs,
        'history': callback.history,
        'n_true_evals': problem.true_eval_count,
        'n_surrogate_evals': problem.surrogate_eval_count
    }
    
    with open(os.path.join(output_dir, 'final_results.pkl'), 'wb') as f:
        pickle.dump(final_results, f)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"COMPLETE: {problem.eval_count} evals ({problem.true_eval_count} true, {problem.surrogate_eval_count} surrogate)")
        print(f"Pareto front: {len(pareto_configs)} solutions")
        print(f"Results saved to: {output_dir}")
    
    return final_results
