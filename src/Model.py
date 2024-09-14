import torch.nn as nn
import torchvision.models as models

class ResnetModel(nn.Module):
    def __init__(self):
        super(ResnetModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # IMDB 평점은 하나의 실수 값이므로 output은 1개

    def forward(self, x):
        return self.resnet(x).float()
