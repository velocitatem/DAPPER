import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
from torchvision import transforms
# let setup a super class for all models that we will define a function to:
# - transform the image to a tensor that can be used by the model
# - run inference on the model
# train the model

class BaseModel():
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = None
        self.transform = None
        
    def transform_image(self, image: Image.Image) -> torch.Tensor:
        pass
    
    def run_inference(self, image: torch.Tensor) -> torch.Tensor:
        pass
    
    def get_model(self) -> nn.Module:
        return self.model
    

# Import model implementations
from classification.training.resnet import ResNetClassifier
from classification.training.cnn import CNNClassifier
from classification.training.lsnet import LSNetClassifier

def get_model(
    model_name: str, 
    num_classes: int, 
    pretrained: bool = True,
    freeze_backbone: bool = False,
    **kwargs
) -> nn.Module:
    """
    Get a model by name.
    
    Args:
        model_name: Name of the model to get
        num_classes: Number of classes to classify
        pretrained: Whether to use pre-trained weights
        freeze_backbone: Whether to freeze the backbone
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        The requested model
    """
    if model_name == 'resnet18':
        model = ResNetClassifier(num_classes=num_classes, trained_model_name='resnet18', pretrained=pretrained, **kwargs)
        return model.model
    elif model_name == 'resnet34':
        model = ResNetClassifier(num_classes=num_classes, trained_model_name='resnet34', pretrained=pretrained, **kwargs)
        return model.model
    elif model_name == 'resnet50':
        model = ResNetClassifier(num_classes=num_classes, trained_model_name='resnet50', pretrained=pretrained, **kwargs)
        return model.model
    elif model_name == 'resnet101':
        model = ResNetClassifier(num_classes=num_classes, trained_model_name='resnet101', pretrained=pretrained, **kwargs)
        return model.model
    elif model_name == 'cnn':
        model = CNNClassifier(num_classes=num_classes, **kwargs)
        return model.model
    elif model_name == 'lsnet_t':
        model = LSNetClassifier(num_classes=num_classes, model_size='t', **kwargs)
        return model.model
    elif model_name == 'lsnet_s':
        model = LSNetClassifier(num_classes=num_classes, model_size='s', **kwargs)
        return model.model
    elif model_name == 'lsnet_b':
        model = LSNetClassifier(num_classes=num_classes, model_size='b', **kwargs)
        return model.model
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
