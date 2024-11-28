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


class AB_2GR3_10(nn.Module):
    
  def __init__(self, num_classes):
    super(AB_2GR3_10,self).__init__()
    
    self.Conv1 =  nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride =1,padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=False))

    self.Res1 = ResBlock3(in_planes = 64)

    self.Act1 = ANDHRA()

    self.Conv21 =  ResBlockP(in_channels=64, out_channels=128, stride=2)

    self.Res21  = ResBlock3(in_planes = 128)
    
    self.Conv22 =  ResBlockP(in_channels=64, out_channels=128, stride=2)

    self.Res22  = ResBlock3(in_planes = 128)
    

    self.Act21 = ANDHRA()
    self.Act22 = ANDHRA()

    self.Conv31 =  ResBlockP(in_channels=128, out_channels=256, stride=2)

    self.Res31  =  ResBlock3(in_planes = 256)
    
    self.Conv32 =  ResBlockP(in_channels=128, out_channels=256, stride=2)

    self.Res32  =  ResBlock3(in_planes = 256)                            
    
    self.Conv33 =  ResBlockP(in_channels=128, out_channels=256, stride=2)

    self.Res33  =  ResBlock3(in_planes = 256)
    
    self.Conv34 =  ResBlockP(in_channels=128, out_channels=256, stride=2)

    self.Res34  =  ResBlock3(in_planes = 256)
    

    self.Act31 = ANDHRA()
    self.Act32 = ANDHRA()
    self.Act33 = ANDHRA()
    self.Act34 = ANDHRA()
  
    self.Conv41 =  ResBlockP(in_channels=256, out_channels=512, stride=2)

    self.Res41  =  ResBlock3(in_planes = 512)
    
    self.Conv42 =  ResBlockP(in_channels=256, out_channels=512, stride=2)

    self.Res42  =  ResBlock3(in_planes = 512)
    
    self.Conv43 =  ResBlockP(in_channels=256, out_channels=512, stride=2)

    self.Res43  =  ResBlock3(in_planes = 512)
    
    self.Conv44 =  ResBlockP(in_channels=256, out_channels=512, stride=2)

    self.Res44  =  ResBlock3(in_planes = 512)
    
    self.Conv45 =  ResBlockP(in_channels=256, out_channels=512, stride=2)

    self.Res45  =  ResBlock3(in_planes = 512)
    
    self.Conv46 =  ResBlockP(in_channels=256, out_channels=512, stride=2)

    self.Res46  =  ResBlock3(in_planes = 512)
    
    self.Conv47 =  ResBlockP(in_channels=256, out_channels=512, stride=2)

    self.Res47  =  ResBlock3(in_planes = 512)
    
    self.Conv48 =  ResBlockP(in_channels=256, out_channels=512, stride=2)
                                
    self.Res48  =  ResBlock3(in_planes = 512)

    self.Relu   =  nn.ReLU(inplace=False)
    
    self.pool4 =   nn.AvgPool2d(kernel_size=4) 
    



    self.Linear1 = nn.Linear(512, num_classes)
    self.Linear2 = nn.Linear(512, num_classes)
    self.Linear3 = nn.Linear(512, num_classes)
    self.Linear4 = nn.Linear(512, num_classes)
    self.Linear5 = nn.Linear(512, num_classes)
    self.Linear6 = nn.Linear(512, num_classes)
    self.Linear7 = nn.Linear(512, num_classes)
    self.Linear8 = nn.Linear(512, num_classes)


    

  def forward(self,x):

    out = self.Res1(self.Conv1(x))


    out1, out2 = self.Act1(out)

    out1 = self.Res21(self.Conv21(out1))
    out2 = self.Res22(self.Conv22(out2))


    out11, out12 = self.Act21(out1)
    out21, out22 = self.Act22(out2)


    out11 = self.Res31(self.Conv31(out11))
    out12 = self.Res32(self.Conv32(out12))
    out21 = self.Res33(self.Conv33(out21))
    out22 = self.Res34(self.Conv34(out22))


    out111, out112 = self.Act31(out11)
    out121, out122 = self.Act32(out12)
    out211, out212 = self.Act33(out21)
    out221, out222 = self.Act34(out22)

    out111 = self.Linear1(self.pool4(self.Relu(self.Res41(self.Conv41(out111)))).view(out.size(0), -1))
    out112 = self.Linear2(self.pool4(self.Relu(self.Res42(self.Conv42(out112)))).view(out.size(0), -1))
    out121 = self.Linear3(self.pool4(self.Relu(self.Res43(self.Conv43(out121)))).view(out.size(0), -1))
    out122 = self.Linear4(self.pool4(self.Relu(self.Res44(self.Conv44(out122)))).view(out.size(0), -1))
    out211 = self.Linear5(self.pool4(self.Relu(self.Res45(self.Conv45(out211)))).view(out.size(0), -1))
    out212 = self.Linear6(self.pool4(self.Relu(self.Res46(self.Conv46(out212)))).view(out.size(0), -1))
    out221 = self.Linear7(self.pool4(self.Relu(self.Res47(self.Conv47(out221)))).view(out.size(0), -1))
    out222 = self.Linear8(self.pool4(self.Relu(self.Res48(self.Conv48(out222)))).view(out.size(0), -1))



    return out111, out112, out121, out122, out211, out212, out221, out222

def ab_2GR3_10():
    net = AB_2GR3_10(10)
    return net
