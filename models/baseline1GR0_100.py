import torch
import torch.nn as nn


class Act(nn.Module):
  def __init__(self):
    super(Act,self).__init__()
    self.Relu1 = nn.ReLU(inplace=False)

  def forward(self,x):

    x1 = self.Relu1(x)

    return x1



class ResBlockP(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlockP, self).__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride= stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        #shortcut
        self.shortcut = nn.Sequential(
              nn.Conv2d(in_channels, out_channels, kernel_size=1, stride= stride, bias=False),
              nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return nn.ReLU(inplace=False)(self.residual_function(x) + self.shortcut(x))



class Baseline1GR0_100(nn.Module):
    
  def __init__(self, num_classes):
    super(Baseline1GR0_100,self).__init__()
    
    self.Conv1 =  nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride =1,padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=False))


    self.Act1 = Act()

    self.Conv21 =  ResBlockP(in_channels=64, out_channels=128, stride=2)
    
    self.Act21 = Act()

    self.Conv31 =  ResBlockP(in_channels=128, out_channels=256, stride=2)

    self.Act31 = Act()
  
    self.Conv41 =  ResBlockP(in_channels= 256, out_channels= 512, stride=2)

    self.Relu = nn.ReLU(inplace=False)
    
    self.pool4 = nn.AvgPool2d(kernel_size=4) 
  
    self.Linear1 = nn.Linear(512, num_classes)


  def forward(self,x):

    out = self.Conv1(x)
    out = self.Act1(out)

    out = self.Conv21(out)
    out  = self.Act21(out)

    out = self.Conv31(out)
    out = self.Act31(out)

    out = self.Linear1(self.pool4(self.Relu(self.Conv41(out))).view(out.size(0), -1))

    return out


def baseline1GR0_100():
    net = Baseline1GR0_100(100)
    return net

