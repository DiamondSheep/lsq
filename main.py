import os 
import torch
import torch.nn as nn
import argparse
import logging

from config_parser import ConfigParser, print_config, benchmark, logger
from model_utils import *
from dataloader import *
from eval_utils import *

parser = argparse.ArgumentParser(description='Quant')
parser.add_argument('--settings', default='./settings_det.hocon', type=str,
                    help='Configuration path')
parser.add_argument('--model_path', default=None, type=str,
                    help='path to load pretrained model')
parser.add_argument('--seed', '-s', default=1, type=int,
                    help='random seed setting')
args = parser.parse_args()
configs = ConfigParser(args.settings)

### GPU device setting 
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device
print("-- CUDA device: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

### Logger setting
log_path = configs.set_logging()
configs.seed = args.seed

if __name__ == "__main__":

    print_config(logger, configs)

    ### Model
    model = get_fp_model(configs.task, configs.dataset, configs.model, args.model_path)
    if model is not None:
        logger.info("model is loaded.")
        model = model.cuda()
    else:
        logger.info("failed to load model.")
        exit(0)

    ### Load validation data
    logger.info(f'Preparing data: {configs.dataset}')
    valloader = getValData(dataset=configs.dataset,
                        batch_size=configs.batch_size,
                        path=configs.datasetPath,
                        img_size=configs.img_size,
                        for_inception=configs.model.startswith('inception'))

    ### Validate original model
    #top1, top5 = validate(model, valloader, logger)
    #logger.info("Full-precision results\n\tTop1:{:.3f}\tTop5:{:.3f}".format(top1, top5))

    ### Validation
    top1, top5 = validate(model, valloader, logger, configs)