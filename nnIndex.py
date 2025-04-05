import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_chnls, classes):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Sequential(         
            nn.Conv2d(in_chnls, 16, 3, 1, 1),                              
            nn.ReLU(),                      
            nn.MaxPool2d(2), 
        )   
        self.conv_layer2 = nn.Sequential(         
            nn.Conv2d(16, 32, 3, 1, 1),                              
            nn.ReLU(),                      
            nn.MaxPool2d(2), 
        )
        self.conv_layer3 = nn.Sequential(         
            nn.Conv2d(32, 64, 3, 1, 1),                              
            nn.ReLU(),
        )
        self.conv_layer4 = nn.Sequential(         
            nn.Conv2d(64, 128, 3, 1, 1),                              
            nn.ReLU(),                      
            nn.MaxPool2d(2),
        )
        self.fc_hidden = nn.Linear(128*4*4, 256)
        self.fc_output = nn.Linear(256,classes)
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = x.view(x.size(0), -1)       
        x = self.fc_hidden(x)
        output = self.fc_output(x)
        return output

class MLP(nn.Module):
    def __init__(self, in_chnls, classes):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * in_chnls, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Linear(256, classes),
        )
    def forward(self, x):
        output = self.fc_layers(x)
        return output
