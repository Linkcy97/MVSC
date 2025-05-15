import torch.nn as nn



class Cla_Net(nn.Module):
    
    def __init__(self, dim):
        super(Cla_Net, self).__init__()
        self.classfiy_net = nn.Sequential(
            nn.Conv2d(dim, dim//2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim//2, 10, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(10, 10))
        
    def forward(self, x):
        return self.classfiy_net(x)