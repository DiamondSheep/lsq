import torch
import logging
# ImageNet
from pytorchcv.model_provider import get_model

from dataloader import *
from eval_utils import *
from config_parser import logger

def get_fp_model(task, dataset, model_name, model_path=None):
    logger.info(f'Loading pretrained model {model_name} for {dataset} dataset for {task} task')
    # For classification
    if (task == 'classification'):
        if (model_path):
            # TODO: from given model path

            model.load_state_dict(torch.load(model_path))
        else:
            # from pytorchcv
            if dataset == 'imagenet':    
                pass
            elif dataset == 'cifar100':
                model = model + '_cifar100'
            elif dataset == 'cifar10':
                model = model + '_cifar10'
            else:
                logger.info(f"No supported dataset {dataset} for task {task}.\n")
                exit(0)
            model = get_model(model_name, pretrained=True)
        return model

    # For detection
    elif (task == 'detection'):
        if (model_path):
            # from given model path
            model = get_model(model_name, pretrained=False)
            model.load_state_dict(torch.load(model_path))
            return model
        elif dataset == 'coco':
            # from torch hub 
            if model_name in ['yolov3', 'yolov5s']: # TODO: add more yolo models
                model = torch.hub.load(f'ultralytics/{model_name}', f'{model_name}')
            else: 
                logger.info(f"No supported model {model_name}")

            return model
        else:
            logger.info(f"No supported dataset {dataset} for task {task}.\n")

    else:
        logger.info(f"No supported model {model_name} for task {task}.\n")
        exit(0)

if __name__ == "__main__":
    pass