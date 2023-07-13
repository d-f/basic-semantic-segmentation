import torchvision
from pathlib import Path
import torch
from typing import Dict


def define_model(pre_trained: bool, num_classes: int) -> torchvision.models.segmentation.fcn_resnet50:
    '''
    defines the model architecture
    '''
    model = torchvision.models.segmentation.fcn_resnet50(
        pretrained=pre_trained, 
        num_classes=num_classes, 
        weights=None
        )
    return model 


def print_model_summary(model: torchvision.models.segmentation.fcn_resnet50) -> None:
    '''
    prints the parameters and parameter size
    '''
    for param in model.named_parameters():
        print(param[0], param[1].size())


def load_model(
        weight_path: Path, 
        model: torchvision.models.segmentation.fcn_resnet50
        ) -> torchvision.models.segmentation.fcn_resnet50:
    '''
    loads all parameters of a model
    '''
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint)
    return model


def define_optimizer(model: torchvision.models.segmentation.fcn_resnet50, learning_rate: float):    
    '''
    returns optimizer
    '''     
    return torch.optim.Adam(params=model.parameters(), lr=learning_rate)


def define_criterion() -> torch.nn.CrossEntropyLoss:
    '''
    returns loss function
    '''
    return torch.nn.CrossEntropyLoss()


def define_device() -> torch.device:
    '''
    returns torch device
    '''
    return torch.device("cuda")


def save_checkpoint(state: Dict, filepath: Path) -> None:
    '''
    saves the model state dictionary to a .pth.tar tile
    '''
    print("saving...")
    torch.save(state, filepath)
