import torch
import torch.nn as nn


class ANDHRA(nn.Module):
  def __init__(self):
    super(ANDHRA,self).__init__()
    self.Relu1 = nn.ReLU(inplace=False)
    self.Relu2 = nn.ReLU(inplace=False)

  def forward(self,x):

    x1 = self.Relu1(x)

    x2 = self.Relu2(x)

    return x1, x2

class ResBlock3(nn.Module):
    def __init__(self, in_planes):
        super(ResBlock3, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=3, stride =1,padding=1, bias=False),
                                nn.BatchNorm2d(in_planes),
                                nn.ReLU(inplace=False),
                                nn.Conv2d(in_planes,in_planes, kernel_size=3, stride =1,padding=1, bias=False),
                                nn.BatchNorm2d(in_planes))
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=3, stride =1,padding=1, bias=False),
                                nn.BatchNorm2d(in_planes),
                                nn.ReLU(inplace=False),
                                nn.Conv2d(in_planes,in_planes, kernel_size=3, stride =1,padding=1, bias=False),
                                nn.BatchNorm2d(in_planes))
        self.conv3 = nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=3, stride =1,padding=1, bias=False),
                                nn.BatchNorm2d(in_planes),
                                nn.ReLU(inplace=False),
                                nn.Conv2d(in_planes,in_planes, kernel_size=3, stride =1,padding=1, bias=False),
                                nn.BatchNorm2d(in_planes))

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1(x)
        out += self.shortcut(x)
        x = self.relu(out)

        out = self.conv2(x)
        out += self.shortcut(x)
        x = self.relu(out)

        out = self.conv3(x)
        out += self.shortcut(x)
        return out


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


class AB_2GR3_2H2_10(nn.Module):
    
  def __init__(self, num_classes):
    super(AB_2GR3_2H2_10,self).__init__()
    
    self.Conv1 =  nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride =1,padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=False))

    self.Res1 = ResBlock3(in_planes = 64)

    self.Act1 = nn.ReLU(inplace=False)

    self.Conv21 =  ResBlockP(in_channels=64, out_channels=128, stride=2)

    self.Res21  = ResBlock3(in_planes = 128)
    

    self.Act21 = ANDHRA()

    self.Conv31 =  ResBlockP(in_channels=128, out_channels=256, stride=2)

    self.Res31  =  ResBlock3(in_planes = 256)
    
    self.Conv32 =  ResBlockP(in_channels=128, out_channels=256, stride=2)

    self.Res32  =  ResBlock3(in_planes = 256)                            
    
    

    self.Act31 = nn.ReLU(inplace=False)
    self.Act32 = nn.ReLU(inplace=False)
  
    self.Conv41 =  ResBlockP(in_channels=256, out_channels=512, stride=2)

    self.Res41  =  ResBlock3(in_planes = 512)
    
    self.Conv42 =  ResBlockP(in_channels=256, out_channels=512, stride=2)

    self.Res42  =  ResBlock3(in_planes = 512)
  

    self.Relu   =  nn.ReLU(inplace=False)
    
    self.pool4 =   nn.AvgPool2d(kernel_size=4) 
    



    self.Linear1 = nn.Linear(512, num_classes)
    self.Linear2 = nn.Linear(512, num_classes)


    

  def forward(self,x):

    out = self.Res1(self.Conv1(x))


    out = self.Act1(out)

    out = self.Res21(self.Conv21(out))


    out1, out2 = self.Act21(out)


    out1 = self.Res31(self.Conv31(out1))
    out2 = self.Res32(self.Conv32(out2))


    out1 = self.Act31(out1)
    out2 = self.Act32(out2)

    out1 = self.Linear1(self.pool4(self.Relu(self.Res41(self.Conv41(out1)))).view(out.size(0), -1))
    out2 = self.Linear2(self.pool4(self.Relu(self.Res42(self.Conv42(out2)))).view(out.size(0), -1))


    return out1, out2

def ab_2GR3_2H2_10():
    net = AB_2GR3_2H2_10(10)
    return net
