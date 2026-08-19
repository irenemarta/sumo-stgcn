import torch
import torch.nn as nn

# TODO: modifica inferenza per integrare velocity field 


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
        t_expanded = t.expand(x_t.shape[0], 1)
        inp = torch.cat([x_t, t_expanded, cond], dim=-1)
        return self.net(inp)
    

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