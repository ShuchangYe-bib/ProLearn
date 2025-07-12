import torch.nn as nn
from transformers import AutoModel

class VisionModel(nn.Module):

    def __init__(self, vision_type):
        super(VisionModel, self).__init__()
        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True)   

    def forward(self, x):
        return self.model(x, output_hidden_states=True)['hidden_states']