import os 
import torch
import torch.nn as nn
import argparse
import logging

from config_parser import ConfigParser, print_config, logger
from model_utils import get_fp_model
from dataloader import getValData, getTrainData
from eval_utils import validate, validate_det, fine_tuning
from quant_utils import quantize_model
from quant_utils.quant_model import freeze_model

parser = argparse.ArgumentParser(description='Quant')
parser.add_argument('--settings', default='./settings_cls.hocon', type=str,
                    help='Configuration path')
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
    model = get_fp_model(configs.task, configs.dataset, configs.model, configs.model_path)

    ### Load validation data
    if configs.task == 'classification':
        logger.info(f'Preparing data: {configs.dataset}')
        valloader = getValData(dataset=configs.dataset,
                            batch_size=configs.batch_size,
                            path=configs.valDatasetPath,
                            img_size=configs.img_size,
                            for_inception=configs.model.startswith('inception'))
        trainloader = getTrainData(dataset=configs.dataset,
                                   batch_size=configs.batch_size,
                                   path=configs.trainDatasetPath,
                                   for_inception=configs.model.startswith('inception'))
    elif configs.task == 'detection':
        pass

    ### Quantization
    quant_model = quantize_model(model, weight_bit=configs.wbit, 
                                        act_bit=configs.abit, 
                                        mode="LSQ")
    logger.info(f"model quantized.")
    
    ### Quantization-Aware Training
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(quant_model.parameters(), 
                                lr=configs.lr,
                                momentum=configs.momentum, 
                                weight_decay=configs.weight_decay)
    
    fine_tuning(quant_model, trainloader, criterion, optimizer, logger, configs)
    quant_model = freeze_model(model, mode="LSQ")

    ### Validation
    if configs.task == 'classification':
        #validate(quant_model, valloader, logger, configs)
        validate(model, valloader, logger, configs)
    elif configs.task == 'detection':
        validate_det(model=model)
    else:
        logger.info("Unsupported task.")