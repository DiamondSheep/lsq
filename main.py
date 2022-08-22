import os 
import torch
import torch.nn as nn
import argparse
import logging

from pytorchcv.model_provider import get_model
from config_parser import ConfigParser, print_config, benchmark
from dataloader import *
from eval_utils import *

parser = argparse.ArgumentParser(description='SQuant')
parser.add_argument('--settings', default='./settings.hocon', type=str,
                    help='Configuration path')
parser.add_argument('--model_path', default=None, type=str,
                    help='path to load pretrained model')
parser.add_argument('--seed', '-s', default=1, type=int,
                    help='random seed setting')
args = parser.parse_args()

configs = ConfigParser(args.settings)

### Setting GPU device
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device
print("-- CUDA device: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

if __name__ == "__main__":
    ### Logger setting
    log_path = configs.set_logging()
    logger = logging.getLogger(__name__)
    configs.seed = args.seed
    print_config(logger, configs)

    ### Model
    logger.info(f'Loading pretrained model: {configs.model}')

    if (args.model_path):
        print("Load local model path: {}".format(args.model_path))
        model = get_model(configs.model, pretrained=False)
        model.load_state_dict(torch.load(args.model_path))
    else:
        model = get_model(configs.model, pretrained=True)
    model = model.cuda()

    ### Load validation data
    logger.info(f'Preparing data: {configs.dataset}')
    valloader = getValData(dataset=configs.dataset,
                        batch_size=configs.batch_size,
                        path=configs.datasetPath,
                        for_inception=configs.model.startswith('inception'))

    ### Validate original model
    #top1, top5 = validate(model, valloader, logger)
    #logger.info("Full-precision results\n\tTop1:{:.3f}\tTop5:{:.3f}".format(top1, top5))

    ### Validation
    top1, top5 = validate(model, valloader, logger)