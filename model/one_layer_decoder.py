import torch.nn as nn

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