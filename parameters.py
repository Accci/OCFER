import numpy as np

### Setting hyperparameters
class hyperparameters():
    def __init__(self):
        self.bs = 64
        self.epoch = 10
        self.lr = 1e-5
        self.sigma = 0.001
        self.sigma1 = 0.0000000000000000000000000000000000000001
        self.D = 512
        self.N = 0.5
        self.gamma = float(1/512.0)
        self.alpha = 1.0

        self.img_chnl = 1
        self.img_size = 48

        self.gpu_flag = True
        self.verbose = False
        self.pre_trained_flag = False
        self.intensity_normalization = False
        self.arch ='CBAM'
        self.classifier_type = 'OC-CNN'


        self.HEADER = '\033[95m'
        self.BLUE= '\033[94m'
        self.GREEN= '\033[92m'
        self.YELLOW= '\033[93m'
        self.FAIL= '\033[91m'
        self.ENDC= '\033[0m'
        self.BOLD= '\033[1m'
        self.UNDERLINE = '\033[4m'


hyper_para = hyperparameters()
