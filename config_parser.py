import os
import logging
import uuid
from pyhocon import ConfigFactory

logger = logging.getLogger(__name__)

class ConfigParser(object):
	def __init__(self, conf_path):
		super(ConfigParser, self).__init__()
		self.conf = ConfigFactory.parse_file(conf_path)

		#  ---------- General settings ----------
		self.device = str(self.conf['device'])
		self.task = str(self.conf['task'])
		self.dataset = self.conf['dataset']
		self.datasetPath = self.conf['dataset_path']  # path for loading data set
		self.img_size = self.conf['img_size']
		self.batch_size = self.conf['batch_size']
		self.model = self.conf['model']
		self.model_path = self.conf['model_path']
		
		# -------- Quantization Settings -------
		self.mode = self.conf['mode']
		self.wbit = self.conf['weight_bit_width']
		self.abit = self.conf['activation_bit_width']

		self.log_path = None
		self.seed = 1

	def set_logging(self, path='quantization_log'):
		if not os.path.isdir(path):
			os.mkdir(path)
		path = os.path.join(path, f"{self.dataset}_{self.model}")
		if not os.path.isdir(path):
			os.mkdir(path)

		num = int(uuid.uuid4().hex[0:4], 16)
		pathname = f"{self.mode}_W{self.wbit}A{self.abit}_{str(num)}"
		path = os.path.join(path, pathname)
		if not os.path.isdir(path):
			os.mkdir(path)
		
		self.log_path = os.path.join(path, "training.log")
		logging.basicConfig(
			format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
			datefmt='%Y%m/%d/ %H:%M:%S',
			level=logging.INFO,
			handlers=[
				logging.FileHandler(self.log_path),
				logging.StreamHandler()
			]
		)
		return path

def print_config(logger, configs):
	logger.info(configs.log_path)
	information = 'Configurations:\n\n'
	for key in vars(configs):
		if (key == 'conf'):
			continue
		information += (f"\t-- {key} : {vars(configs)[key]}\n")
	logger.info(information)

def benchmark(log_path):
	file_name = "benchmark.txt"
	file_name = os.path.join(log_path, file_name)
	benchmark_file = open(file_name, 'w')
	return 