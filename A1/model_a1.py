
import torch.nn as nn

class ModelA1(nn.Module):
    """
    CNN is used for task A1, with multiple layers
    """
    def __init__(self, num_classes = 1) -> None:
        super().__init__()
        self.num_classes = num_classes
        #Layers of CNN model
        self.layers = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),#32,32
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),#16,16
            nn.Conv2d(64,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        #linear layer
        self.linear = nn.Sequential(
            nn.Linear(16*16*16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_classes)
        )
    def forward(self, x):
        """
        Forward images 
        Input:
            Preprocessed image
        Return:
            Output of model
        """
        x = self.layers(x)
        x = x.flatten(1)
        x = self.linear(x)
        return x
    

