import os
import pandas as pd
import json
from classification.utils.logger import get_standard_logger

# Get standard logger directly
logger = get_standard_logger("cache_loader")
DIR_CACHE = "data_cache"

def json_to_string(params: dict):
    vals = []
    for k, v in params.items():
        if isinstance(v, bool):
            vals.append(f"{k}={v}")
        else:
            vals.append(f"{k}={v}")
    return "_".join(vals)

def load_dataset_from_cache(dataset_name: str, params: dict):
    params_string = json_to_string(params)
    cache_key = f"{dataset_name}_{params_string}"
    cache_file = f"{DIR_CACHE}/{cache_key}.pkl"
    if os.path.exists(cache_file):
        logger.info(f"Loading cached dataset from {cache_file}")
        return pd.read_pickle(cache_file)
    else:
        logger.info(f"No cache found at {cache_file}")
        return None

def save_dataset_to_cache(dataset_name: str, params: dict, df: pd.DataFrame = None):
    params_string = json_to_string(params)
    cache_key = f"{dataset_name}_{params_string}"
    
    # Create cache directory if it doesn't exist
    os.makedirs(DIR_CACHE, exist_ok=True)
    
    cache_file = f"{DIR_CACHE}/{cache_key}.pkl"
    if os.path.exists(cache_file):
        logger.info(f"Removing existing cache file {cache_file}")
        os.remove(cache_file)
    
    if df is None:
        logger.warning(f"No DataFrame provided to save in cache file {cache_file}")
        return False
        
    try:
        logger.info(f"Saving dataset to cache {cache_file}")
        df.to_pickle(cache_file)
        return True
    except Exception as e:
        logger.error(f"Failed to save cache file {cache_file}: {str(e)}")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        return False
