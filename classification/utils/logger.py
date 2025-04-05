import os
import logging
import time
from typing import Dict, Any, Optional
import yaml
from torch.utils.tensorboard import SummaryWriter

# Configure the global logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def get_standard_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a standard Python logger configured with consistent formatting.
    This is a simpler alternative to the Logger class when TensorBoard isn't needed.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        A configured standard Python logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Console handler with standard format
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

class Logger:
    """
    Advanced logger utility for both TensorBoard and standard logging.
    Handles experiment tracking, metrics, and config logging.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        enable_tensorboard: bool = True
    ):
        """
        Initialize logger
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment (defaults to timestamp if None)
            config: Configuration dictionary to save
            console_level: Logging level for console output
            file_level: Logging level for file output
            enable_tensorboard: Whether to enable TensorBoard logging
        """
        # Create experiment name if not provided
        if experiment_name is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            experiment_name = f"experiment_{timestamp}"
            
        # Create log directory
        self.log_dir = os.path.join(log_dir, experiment_name)
        self.tensorboard_dir = os.path.join(self.log_dir, "tensorboard")
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if enable_tensorboard and not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
            
        # Set up file logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers if any
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
            
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # File handler
        log_file = os.path.join(self.log_dir, "experiment.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Initialize TensorBoard writer if enabled
        self.enable_tensorboard = enable_tensorboard
        if enable_tensorboard:
            self.writer = SummaryWriter(self.tensorboard_dir)
        else:
            self.writer = None
        
        # Save config if provided
        if config is not None:
            self.save_config(config)
            
        self.logger.info(f"Logger initialized: {experiment_name}")
        self.logger.info(f"Log directory: {self.log_dir}")
            
    def log_metrics(self, metrics: Dict[str, float], step: int = None, prefix: str = ""):
        """
        Log metrics to both TensorBoard and log file
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step/epoch number
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        for name, value in metrics.items():
            metric_name = f"{prefix}{name}" if prefix else name
            
            if step is not None:
                if self.enable_tensorboard and self.writer:
                    self.writer.add_scalar(metric_name, value, step)
                self.logger.info(f"{metric_name} at step {step}: {value:.6f}")
            else:
                self.logger.info(f"{metric_name}: {value:.6f}")
                
    def log_histogram(self, name: str, values, step: int):
        """Log histogram to TensorBoard"""
        if self.enable_tensorboard and self.writer:
            self.writer.add_histogram(name, values, step)
        
    def log_images(self, name: str, images, step: int, dataformats: str = 'NCHW'):
        """Log images to TensorBoard"""
        if self.enable_tensorboard and self.writer:
            self.writer.add_images(name, images, step, dataformats=dataformats)
        
    def log_model_graph(self, model, input_size=(1, 3, 224, 224)):
        """Log model graph to TensorBoard"""
        if self.enable_tensorboard and self.writer:
            import torch
            device = next(model.parameters()).device
            dummy_input = torch.zeros(input_size, device=device)
            self.writer.add_graph(model, dummy_input)
        
    def save_config(self, config: Dict[str, Any], filename: str = "config.yaml"):
        """Save configuration to YAML file"""
        config_path = os.path.join(self.log_dir, filename)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        self.logger.info(f"Configuration saved to {config_path}")
        
    def close(self):
        """Close the TensorBoard writer"""
        if self.enable_tensorboard and self.writer:
            self.writer.close()
        
    def __del__(self):
        """Ensure writer is closed when object is destroyed"""
        try:
            self.close()
        except:
            pass
            
# Convenience function to get a logger
def get_logger(name: Optional[str] = None, config: Optional[Dict[str, Any]] = None, 
              enable_tensorboard: bool = False) -> Logger:
    """
    Get a configured Logger instance.
    For simple cases without TensorBoard, consider using get_standard_logger instead.
    
    Args:
        name: Experiment name
        config: Configuration dictionary
        enable_tensorboard: Whether to enable TensorBoard
    
    Returns:
        A Logger instance
    """
    return Logger(experiment_name=name, config=config, enable_tensorboard=enable_tensorboard) 