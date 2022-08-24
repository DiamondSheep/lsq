import torch
import logging
from pytorchcv.model_provider import get_model
from dataloader import *
from eval_utils import *
from config_parser import logger

def get_fp_model(task, dataset, model_name, model_path=None):
    logger.info(f'Loading pretrained model {model_name} for {dataset} dataset for {task} task')
    if (task == 'classification'):
        if (model_path):
                # from given model path
                model = get_model(model_name, pretrained=False)
                model.load_state_dict(torch.load(model_path))

        else:
            if dataset == 'imagenet':    
                pass
            elif dataset == 'cifar100':
                model = model + '_cifar100'
            elif dataset == 'cifar10':
                model = model + '_cifar10'
            else:
                logger.info(f"No supported dataset {dataset} for task {task}.\n")
                exit(0)
            
            # from pytorchcv
            model = get_model(model_name, pretrained=True)

        return model

    elif (task == 'detection'):
        if dataset == 'coco':
            # TODO: load coco dataset here
            logger.info("to load coco data here.")
            exit(0)
            pass
        else:
            logger.info(f"No supported dataset {dataset} for task {task}.\n")
        
    else:
        logger.info(f"No supported model {model_name} for task {task}.\n")
        exit(0)

if __name__ == "__main__":
    pass