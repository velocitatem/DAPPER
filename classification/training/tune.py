import optuna
import yaml
import argparse
import sys
import os
import copy
from classification.training.train import run_training, load_config # Import the refactored function and config loader
from classification.utils.logger import get_standard_logger
from typing import Dict, Any

logger = get_standard_logger("tune")

def get_value_from_trial(trial: optuna.trial.Trial, param_config: Dict[str, Any]):
    """Suggests a value for a parameter based on its configuration."""
    param_type = param_config['type']
    
    if param_type == 'categorical':
        return trial.suggest_categorical(name=param_config['name'], choices=param_config['choices'])
    elif param_type == 'float':
        return trial.suggest_float(
            name=param_config['name'], 
            low=param_config['low'], 
            high=param_config['high'], 
            log=param_config.get('log', False) # Use log scale if specified
        )
    elif param_type == 'int':
        return trial.suggest_int(
            name=param_config['name'], 
            low=param_config['low'], 
            high=param_config['high'],
            log=param_config.get('log', False)
        )
    else:
        raise ValueError(f"Unsupported parameter type: {param_type}")

def set_nested_value(d: Dict[str, Any], keys: str, value: Any):
    """Sets a value in a nested dictionary using a dot-separated key string."""
    key_list = keys.split('.')
    current_level = d
    for i, key in enumerate(key_list):
        if i == len(key_list) - 1:
            current_level[key] = value
        else:
            current_level = current_level.setdefault(key, {}) # Create dict if intermediate key doesn't exist

def objective(trial: optuna.trial.Trial, base_config: Dict[str, Any]) -> float:
    """Objective function for Optuna to minimize/maximize."""
    
    # Create a deep copy of the base config for this trial to avoid interference
    trial_config = copy.deepcopy(base_config)
    
    # --- Suggest hyperparameters based on the 'tuning' section ---
    tuning_params = trial_config.get('tuning', {}).get('params', {})
    if not tuning_params:
        logger.warning("No 'tuning.params' section found in the config. Running with base config.")
    else:
        logger.info(f"Trial {trial.number}: Suggesting parameters...")
        for param_name_dot, param_details in tuning_params.items():
            # Add name to details for suggest function
            param_details['name'] = param_name_dot 
            try:
                suggested_value = get_value_from_trial(trial, param_details)
                # Set the suggested value in the nested trial_config
                set_nested_value(trial_config, param_name_dot, suggested_value)
                logger.info(f"  Suggested {param_name_dot}: {suggested_value}")
            except Exception as e:
                 logger.error(f"Error processing parameter {param_name_dot}: {e}")
                 # Optionally re-raise or return a failure value like float(\'inf\') or float(\'-inf\')
                 raise # Re-raise to stop the study if config is broken
            
    # --- Modify experiment name for logging this specific trial ---
    base_experiment_name = trial_config.get('logging', {}).get('experiment_name', 'tuning_run')
    trial_config.setdefault('logging', {})['experiment_name'] = f"{base_experiment_name}_trial_{trial.number}"
    
    # --- Run the training with the suggested config ---
    try:
        # Ensure the modified config is passed
        validation_metric = run_training(trial_config) 
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with exception: {e}", exc_info=True)
        # Optuna handles exceptions by default, marking the trial as failed. 
        # You could also return a specific value indicating failure, e.g., 0.0 or float('-inf')
        # depending on optimization direction. Let Optuna handle it for now.
        raise optuna.TrialPruned() # Prune if the trial crashes during training

    # --- Return the metric Optuna should optimize ---
    # Assuming run_training returns the best validation accuracy (higher is better)
    return validation_metric 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning using Optuna")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the base configuration file (e.g., configs/cnn_config.yaml)")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Number of trials to run. Overrides n_trials in config if provided.")
    parser.add_argument("--study-name", type=str, default=None,
                        help="Name for the Optuna study. Defaults to model name from config.")
    parser.add_argument("--storage", type=str, default=None,
                        help="Database URL for Optuna storage (e.g., sqlite:///tuning.db). If None, uses in-memory storage.")

    args = parser.parse_args()

    # --- Load Base Config ---
    try:
        base_config = load_config(args.config)
        logger.info(f"Loaded base configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load base config file {args.config}: {e}")
        sys.exit(1)

    # --- Determine Study Details ---
    model_name = base_config.get('model', {}).get('name', 'unknown_model')
    study_name = args.study_name if args.study_name else f"{model_name}_study"
    
    n_trials = args.n_trials
    if n_trials is None:
         n_trials = base_config.get('tuning', {}).get('n_trials', 20) # Default to 20 if not in config
    
    storage = args.storage
    if storage:
        logger.info(f"Using Optuna storage: {storage}")
    else:
        logger.info("Using in-memory Optuna storage (results lost on exit)")

    # --- Create and Run Study ---
    try:
        # We want to maximize validation accuracy
        study = optuna.create_study(
            study_name=study_name, 
            storage=storage,
            direction='maximize', 
            load_if_exists=True # Resume study if storage and name match
        ) 
        
        # Pass the base config to the objective function via a lambda
        objective_with_config = lambda trial: objective(trial, base_config)

        logger.info(f"Starting Optuna study '{study_name}' for {n_trials} trials...")
        study.optimize(objective_with_config, n_trials=n_trials)

        logger.info(f"Study '{study_name}' finished.")
        logger.info(f"Number of finished trials: {len(study.trials)}")
        
        best_trial = study.best_trial
        logger.info(f"Best trial value (validation metric): {best_trial.value}")
        logger.info("Best parameters found:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")
            
    except Exception as e:
        logger.error(f"An error occurred during the Optuna study: {e}", exc_info=True)
        sys.exit(1)

    sys.exit(0) 