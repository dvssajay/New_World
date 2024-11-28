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


class AB_2GR0_10(nn.Module):
    
  def __init__(self, num_classes):
    super(AB_2GR0_10,self).__init__()
    
    self.Conv1 =  nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride =1,padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=False))


    self.Act1 = ANDHRA()

    self.Conv21 =  ResBlockP(in_channels=64, out_channels=128, stride=2)

    
    self.Conv22 =  ResBlockP(in_channels=64, out_channels=128, stride=2)

    self.Act21 = ANDHRA()
    self.Act22 = ANDHRA()

    self.Conv31 =  ResBlockP(in_channels=128, out_channels=256, stride=2)
    
    self.Conv32 =  ResBlockP(in_channels=128, out_channels=256, stride=2)                          
    
    self.Conv33 =  ResBlockP(in_channels=128, out_channels=256, stride=2)
    
    self.Conv34 =  ResBlockP(in_channels=128, out_channels=256, stride=2)
    

    self.Act31 = ANDHRA()
    self.Act32 = ANDHRA()
    self.Act33 = ANDHRA()
    self.Act34 = ANDHRA()
  
    self.Conv41 =  ResBlockP(in_channels=256, out_channels=512, stride=2)

    self.Conv42 =  ResBlockP(in_channels=256, out_channels=512, stride=2)
    
    self.Conv43 =  ResBlockP(in_channels=256, out_channels=512, stride=2)
    
    self.Conv44 =  ResBlockP(in_channels=256, out_channels=512, stride=2)
    
    self.Conv45 =  ResBlockP(in_channels=256, out_channels=512, stride=2)
    
    self.Conv46 =  ResBlockP(in_channels=256, out_channels=512, stride=2)
    
    self.Conv47 =  ResBlockP(in_channels=256, out_channels=512, stride=2)
    
    self.Conv48 =  ResBlockP(in_channels=256, out_channels=512, stride=2)

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

    out = self.Conv1(x)


    out1, out2 = self.Act1(out)

    out1 = self.Conv21(out1) 
    out2 = self.Conv22(out2)

    out11, out12 = self.Act21(out1)
    out21, out22 = self.Act22(out2)

    out11 = self.Conv31(out11)
    out12 = self.Conv32(out12)
    out21 = self.Conv33(out21)
    out22 = self.Conv34(out22)


    out111, out112 = self.Act31(out11)
    out121, out122 = self.Act32(out12)
    out211, out212 = self.Act33(out21)
    out221, out222 = self.Act34(out22)

    out111 = self.Linear1(self.pool4(self.Relu(self.Conv41(out111))).view(out.size(0), -1))
    out112 = self.Linear2(self.pool4(self.Relu(self.Conv42(out112))).view(out.size(0), -1))
    out121 = self.Linear3(self.pool4(self.Relu(self.Conv43(out121))).view(out.size(0), -1))
    out122 = self.Linear4(self.pool4(self.Relu(self.Conv44(out122))).view(out.size(0), -1))
    out211 = self.Linear5(self.pool4(self.Relu(self.Conv45(out211))).view(out.size(0), -1))
    out212 = self.Linear6(self.pool4(self.Relu(self.Conv46(out212))).view(out.size(0), -1))
    out221 = self.Linear7(self.pool4(self.Relu(self.Conv47(out221))).view(out.size(0), -1))
    out222 = self.Linear8(self.pool4(self.Relu(self.Conv48(out222))).view(out.size(0), -1))



    return out111, out112, out121, out122, out211, out212, out221, out222

def ab_2GR0_10():
    net = AB_2GR0_10(10)
    return net
