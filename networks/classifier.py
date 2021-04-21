from torch import nn
import torch
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, backbone):
        super(Net, self).__init__()
        self.backbone = backbone
        self.embedding_size = self.backbone.embedding_size
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



