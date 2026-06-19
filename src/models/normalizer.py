import torch

class MapNormalizer:
    """Normalisation based on standard z-score.
    input: features tensor of shape [N_elements, N_features] --> shape = [2530 x 4]"""
    def __init__(self, map_features: torch.Tensor):
        self.mu = torch.mean(map_features, dim=0, keepdim=True)
        # sqrt(sum(x(i) - mu)^2 / N) 
        self.sigma = torch.sqrt(torch.sum((map_features - self.mu) ** 2, dim=0, keepdim=True) / map_features.shape[0])
    
    def _zscore(self, map: torch.Tensor):
        return ((map - self.mu) / self.sigma)