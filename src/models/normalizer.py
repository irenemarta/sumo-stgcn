import torch

class MapNormalizer:
    """Normalisation based on standard z-score.
    input: features tensor of shape [N_elements, N_features] --> shape = [2530 x 4]"""
    def __init__(self, map_features: torch.Tensor, eps: float = 1e-8):
        self.eps = eps
        reduce_dims = tuple(range(map_features.dim() - 1))
        n = map_features.numel() // map_features.shape[-1]
        self.mu = torch.mean(map_features, dim=reduce_dims, keepdim=True)
        # sqrt(sum(x(i) - mu)^2 / N) 
        self.sigma = torch.sqrt(torch.sum((map_features - self.mu) ** 2, dim=reduce_dims, keepdim=True) / n)
    
    def zscore(self, map_features: torch.Tensor) -> torch.Tensor:
        return (map_features - self.mu) / (self.sigma + self.eps)

    def inverse(self, normalized: torch.Tensor) -> torch.Tensor:
        """Denormalizza — serve per MAE/R2 in unità reali dopo l'integrazione ODE."""
        return normalized * (self.sigma + self.eps) + self.mu

    def save(self, path: str):
        torch.save({"mu": self.mu, "sigma": self.sigma, "eps": self.eps}, path)

    @classmethod
    def load(cls, path: str) -> "MapNormalizer":
        data = torch.load(path)
        obj = cls.__new__(cls)
        obj.mu, obj.sigma, obj.eps = data["mu"], data["sigma"], data["eps"]
        return obj