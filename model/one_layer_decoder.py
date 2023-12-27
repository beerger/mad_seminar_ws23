import torch.nn as nn

# TODO:         # Feature map from Local-Net has size 1 x 1 x 128
        # Feature map from ResNet-18 has size 1 x 1 x 512

class OneLayerDecoder(nn.Module):
    """
    A simple one-layer decoder class that can be used to decode feature vectors
    from one dimension to another. 
    """
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.decoder = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.decoder(x)