import torch
import logging
# ImageNet
from pytorchcv.model_provider import get_model

from dataloader import *
from eval_utils import *
from config_parser import logger
from yolov3.models.common import DetectMultiBackend

def get_fp_model(task, dataset, model_name, model_path=None):

    logger.info(f'Loading pretrained model {model_name} for {dataset} dataset for {task} task')
    
    # For classification
    if (task == 'classification'):
        if (os.path.isfile(model_path)):
            # from given model path
            logger.info(f"load model weight from {model_path}")
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

    # For detection
    elif (task == 'detection'):
        if dataset == 'coco':
            # Load model
            if model_path:
                model = DetectMultiBackend(model_path, device='cuda:0', dnn=False)
                model.model.float()
            else:
                logger.info(f"Illegal model path {model_path}.\n")
                exit(0)
        else:
            logger.info(f"No supported dataset {dataset} for task {task}.\n")
            exit(0)
    else:
        logger.info(f"No supported model {model_name} for task {task}.\n")
        exit(0)

    # return model
    if model is not None:
        logger.info("model is loaded.")
        return model
    else:
        logger.info("failed to load model.")
        exit(0)

if __name__ == "__main__":
    pass