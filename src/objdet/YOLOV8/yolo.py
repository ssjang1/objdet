import torch
import torch.nn as nn
import torch.nn.functional as F

from objdet.YOLOV8.nn.modules.conv import Conv, Concat
from objdet.YOLOV8.nn.modules.block import C2f, SPPF
from nn.modules.head import Detect

class YOLOv8(nn.Module):
    def __init__(self):
        super().__init__()
        self.h0 = Conv(3,64,3,2)
        
        self.detect = DDetect(ch=(256,512,512))
    
    def forward(self, x):

        result = self.detect([x15,x18,x21])
        
        return result