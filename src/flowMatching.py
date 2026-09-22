import torch
import torch.nn as nn
from src.models.mapEncoder import SpatioTemporalConditioner


"""
For a first trial, flow matching has been selected since it is faster than diffusion model denoising process,
still ensuring good quality of prediction. 
Indeed, flow matching is a diffusion process in which the noise is assumed to be Gaussian.


See https://github.com/harshm121/Diffusion-v-FlowMatching/blob/main/comparison_animation.gif

As a reference for the sudy, the following links have been used:
https://diffusion.csail.mit.edu/2026/index.html MIT introductive class about those topics
https://arxiv.org/pdf/2506.02070? course reference notes
https://harshm121.medium.com/flow-matching-vs-diffusion-79578a16c510 Medium introduction to the topics
https://diffusionflow.github.io/ some other overview
"""

# Transformer output = vector field input

class ODE:
    """Velocity field to integrate."""
    def drift_coefficient(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - x_t: state at time t, shape (bs, dim)
            - t: time, shape (batch_size, 1)
            - cond; flow matching conditioner
        Returns:
            - drift_coefficient: shape (batch_size, dim)
        """
        pass

class VelocityVectorField(nn.Module):
    """
    Predicts the velocity fied v(x_t, t, cond) to perform flow matching.
    x_t: features state [N_nodes, N_feat_dyn]
    t: time of interpolation (included in the interval [0,1])
    cond: embedding of [N_nodes, hidden_dims]
    """
    def __init__(self, feat_dyn_dim: int, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dyn_dim + 1 + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dyn_dim),
        )

    def forward(self, x_t, t, cond):
        B, N, _ = cond.shape
        t_expanded = t.view(1, 1, 1).expand(B, N, 1)
        inp = torch.cat([x_t, t_expanded, cond], dim=-1)
        return self.net(inp)
    

class VelocityFieldODE(ODE):
    """Adapts the VeocityVectorField to the ODE."""
    def __init__(self, velocity_fied: VelocityVectorField):
        self.velocity_field = velocity_fied
        
    def drift_coefficient(self, x_t, t, cond):
        return self.velocity_field(x_t, t, cond)


class Simulator:
    def step(self, x_t: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, cond: torch.Tensor):
        """
        Takes one simulation step
        Args:
            - x_t: state at time t, shape (bs, dim)
            - t: time, shape (bs,1)
            - dt: time, shape (bs,1)
            - cond: registered state for each node, in n_hidden parameters
        Returns:
            - nxt: state at time t + dt (bs, dim)
        """
        pass

    @torch.no_grad()
    def simulate(self, x0: torch.Tensor, cond: torch.Tensor, n_integration_steps: int = 50) -> torch.Tensor:
        """
        Simulates using the discretization gives by ts
        Args:
            - x_init: initial state at time ts[0], shape (batch_size, dim)
            - ts: timesteps, shape (bs, num_timesteps,1)
        Returns:
            - x_final: final state at time ts[-1], shape (batch_size, dim)
        """
        h = 1.0 / n_integration_steps
        x_t = x0
        for i in range(n_integration_steps):
            t = torch.tensor([i * h], device=x0.device)
            x_t = self.step(x_t, t, h, cond)
        return x_t


class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode
        
    def step(self, x_t: torch.Tensor, t: torch.Tensor, h: torch.Tensor, cond):
        return x_t + self.ode.drift_coefficient(x_t, t, cond) * h
    
    @torch.no_grad()
    def simulate(self, x0: torch.Tensor, cond: torch.Tensor, n_integration_steps: int = 50) -> torch.Tensor:
        h = 1.0 / n_integration_steps
        x_t = x0
        for i in range(n_integration_steps):
            t = torch.tensor([i * h], device=x0.device)
            x_t = self.step(x_t, t, h, cond)
        return x_t



# conditional flow matching loss
def cfm_loss(velocity_field: VelocityVectorField, x1_target, cond):
    """
    x1_target: target which predicts x_dynamic at t+1 of dimension [N_nodes, N_feat_dyn]
    cond: embedding [N_nodes, hidden_dims]
    """
    noise = torch.randn_like(x1_target) # add random noise with the same dimension of the target
    t = torch.rand(1, device=x1_target.device) # sampled time
    x_t = (1-t)*noise + t*x1_target # linear interpolation between data and noise
    target_vel = x1_target - noise # v on conditional path
    
    pred_vel = velocity_field(x_t, t, cond)
    return torch.mean((pred_vel - target_vel) ** 2)


class FlowMatchingModel(nn.Module):
    def __init__(self, conditioner: SpatioTemporalConditioner, velocity_field: VelocityVectorField):
        super().__init__()
        self.conditioner = conditioner
        self.velocity_field = velocity_field
        
"""
BUILDING BLOCKS:

The Drift Coefficient / Velocity Field:
is the specific speed and direction each grain of sand needs to travel at any given split-second. 
[1] (https://www.youtube.com/watch?v=3mFNpeJQjmw&vl=it), [2] (https://pub.towardsai.net/physics-inspired-generative-modeling-diffusion-flow-matching-and-energy-based-models-9cbacb24488a)

The Conditioner:
is the blueprint or external rule (like a text prompt or class label) telling the sand what kind of structure to build. 
[1] (https://arxiv.org/html/2508.09156v3), [2] (https://arxiv.org/html/2509.19300v1)

The Flow Matching Loss:
is the penalty score measuring how badly a grain of sand went off-course compared to its perfect, straight-line path.
[1] (https://pub.towardsai.net/physics-inspired-generative-modeling-diffusion-flow-matching-and-energy-based-models-9cbacb24488a), [2] (https://scoste.fr/posts/flowmatching/)
"""

@torch.no_grad()
def predict(model: FlowMatchingModel, graphs: list, feat_dyn_dim: int,
            n_integration_steps: int = 50, n_samples: int = 1) -> torch.Tensor:
    """
    Generates n_samples predictions of x' (x_dynamic at t+1), integrating the noise.
    Returns a tensor of shape [n_samples, N_nodes, feat_dyn_dim].
    """
    model.eval()
    cond = model.conditioner(graphs) # [1, N_nodes, hidden] (1=B)
    B, n_nodes, _ = cond.shape
    device = cond.device

    simulator = EulerSimulator(VelocityFieldODE(model.velocity_field))

    predictions = []
    for _ in range(n_samples):
        x0 = torch.randn(B, n_nodes, feat_dyn_dim, device=device) # [B, N_nodes, N_fea_dyn]
        predictions.append(simulator.simulate(x0, cond, n_integration_steps=n_integration_steps))

    return torch.stack(predictions, dim=0) # [n_samples, 1, N_nodes, F], 1 = B