import numpy as np
import torch

from torch.autograd import Variable
from tqdm import tqdm
from Modules import Module


class BatchNormalization(Module):
    EPS = 1e-3
    def __init__(self, alpha = 0.):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None
        self.moving_variance = None

    def updateOutput(self, input):
        # Your code goes here. ################################################
        # use self.EPS please
        return self.output

    def updateGradInput(self, input, gradOutput):
        # Your code goes here. ################################################
        return self.gradInput

    def getParameters(self):
        return [self.moving_mean, self.moving_variance]

    def setParameters(self, parameters):
        self.moving_mean = parameters[0]
        self.moving_variance = parameters[1]

    def __repr__(self):
        return "BatchNormalization"