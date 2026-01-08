import copy
import torch.nn as nn
import torch
import torch.nn.functional as F
import wandb
from utils import global_state


def init_weights_he(module):
    """He/Kaiming initialization for Conv2d and Linear layers with ReLU."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


def init_projection_head(module):
    """Xavier initialization for projection head (no ReLU after final layer)."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, (nn.LayerNorm,)):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


@torch.no_grad()
def momentum_update(online: nn.Module, target: nn.Module, m: float):
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data = p_t.data * m + p_o.data * (1.0 - m)


class MoCo(nn.Module):
    def __init__(
        self,
        backbone,
        proj_dim=128,
        hidden_dim=2048,
        queue_size=65536,
        momentum=0.999,
        temperature=0.2,
    ):
        super(MoCo, self).__init__()
        self.m = momentum
        self.T = temperature
        self.queue_size = queue_size

        self.encoder_q = backbone
        # Create separate encoder_k to avoid shared parameters
        self.encoder_k = copy.deepcopy(backbone)

        self.in_features = backbone.fc.in_features
        self.encoder_k.fc = nn.Identity()
        self.encoder_q.fc = nn.Identity()

        # Projection head with layer norm
        self.proj_q = nn.Sequential(
            nn.Linear(self.in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.proj_k = nn.Sequential(
            nn.Linear(self.in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        # Initialize all learnable parameters with proper initialization
        for encoder in [self.encoder_q, self.encoder_k]:
            encoder.apply(init_weights_he)
        # Projection head uses Xavier for stability (no ReLU after final layer)
        self.proj_q.apply(init_projection_head)
        self.proj_k.apply(init_projection_head)

        # Initialize the key encoder parameters to be the same as query encoder
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer(
            "queue", F.normalize(torch.randn(proj_dim, queue_size), dim=0)
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = keys.detach()
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr.item())
        assert (
            self.queue_size % batch_size == 0
        ), "Queue size must be divisible by batch size"

        end = ptr + batch_size
        if end <= self.queue_size:
            self.queue[:, ptr:end] = keys.T
        else:
            first_part = self.queue_size - ptr
            self.queue[:, ptr : self.queue_size] = keys[:first_part].T
            self.queue[:, 0 : end - self.queue_size] = keys[first_part:].T

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, x_q, x_k):
        q = self.encoder_q(x_q)
        q = self.proj_q(q)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            momentum_update(self.encoder_q, self.encoder_k, self.m)
            momentum_update(self.proj_q, self.proj_k, self.m)

            k = self.encoder_k(x_k)
            k = self.proj_k(k)
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        # Monitor feature statistics
        q_norm = torch.norm(q, dim=1).mean().item()
        k_norm = torch.norm(k, dim=1).mean().item()
        q_std = q.std(dim=0).mean().item()  # Feature diversity

        # Monitor similarity margin (gap between positive and negative)
        margin = (l_pos.mean() - l_neg.mean()).item()

        wandb.log(
            {
                "debug/pos_sim_mean": l_pos.mean().item(),
                "debug/pos_sim_std": l_pos.std().item(),
                "debug/neg_sim_mean": l_neg.mean().item(),
                "debug/neg_sim_std": l_neg.std().item(),
                "debug/q_norm": q_norm,
                "debug/k_norm": k_norm,
                "debug/q_feature_diversity": q_std,
                "debug/similarity_margin": margin,
            },
            step=global_state.get(),
        )

        self._dequeue_and_enqueue(k)
        return logits, labels
