
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel


class EnsembleModel(nn.Module):
    def __init__(self, modelTeacher, modelStudent):
        super(EnsembleModel, self).__init__()
        if isinstance(modelTeacher, (DistributedDataParallel, DataParallel)):
            modelTeacher = modelTeacher.module
        if isinstance(modelStudent, (DistributedDataParallel, DataParallel)):
            modelStudent = modelStudent.module

        self.modelTeacher = modelTeacher
        self.modelStudent = modelStudent

