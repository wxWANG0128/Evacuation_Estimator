import torch.nn as nn
import torch

class LinearNET(nn.Module):
    def __init__(self):
        super(LinearNET,self).__init__()
        self.features = nn.Sequential(
            nn.Linear(6,1,bias=True,dtype=torch.float64)
        )

    def forward(self,x):
        x = self.features(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()


class NeuralNET(nn.Module):
    def __init__(self):
        super(NeuralNET, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(6, 10, bias=True, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(10, 10, bias=True, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(10, 1, bias=True, dtype=torch.float64)
        )

    def forward(self, x):
        x = self.features(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.1)
                m.bias.data.zero_()