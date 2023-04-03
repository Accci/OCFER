import numpy as np

### Setting hyperparameters
class hyperparameters():
    def __init__(self):
        self.bs = 64
        self.epoch = 10
        self.gpu_flag = True
        self.target_class = 0
        self.arch ='CBAM'



        self.HEADER = '\033[95m'
        self.BLUE= '\033[94m'
        self.GREEN= '\033[92m'
        self.YELLOW= '\033[93m'
        self.FAIL= '\033[91m'
        self.ENDC= '\033[0m'
        self.BOLD= '\033[1m'
        self.UNDERLINE = '\033[4m'


hyper_para = hyperparameters()
