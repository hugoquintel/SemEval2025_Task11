from torch import nn

# @title Classification layers
class ClassificationLayers(nn.Module):
    def __init__(self, config, labels_to_ids):
        super().__init__()
        self.cls_layers = nn.Sequential(
                nn.Linear(config.hidden_size, len(labels_to_ids))
        )
    def forward(self, input):
        logit = self.cls_layers(input)
        return logit