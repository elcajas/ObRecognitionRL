import torch
import torch.nn as nn

from mineclip import SimpleFeatureFusion
from mineclip.mineagent.actor.distribution import MultiCategorical
from mineclip.utils import build_mlp

from agents.utils import build_vision_model, predict
from agents import features_mlp as F

class Actor(nn.Module):
    def __init__(
            self,
            *,
            input_dim: int,
            action_dim: list[int],
            hidden_dim: int,
            hidden_depth: int,
            activation: str = "relu",
            deterministic_eval: bool = True,
            device,
    ) -> None:
        super().__init__()
        self.mlps = nn.ModuleList()
        for action in action_dim:
            net = build_mlp(
                input_dim=input_dim,
                output_dim=action,
                hidden_dim=hidden_dim,
                hidden_depth=hidden_depth,
                activation=activation,
                norm_type=None,
            )
            self.mlps.append(net)
        
        self._action_dim = action_dim
        self._device = device
        self._deterministic_eval = deterministic_eval
    
    def forward(self, x) -> torch.Tensor:
        hidden = None
        return torch.cat([mlp(x) for mlp in self.mlps], dim=1), hidden

    @property
    def dist_fn(self):
        return lambda x: MultiCategorical(logits=x, action_dims=self._action_dim)

class Critic(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str = "relu",
        device
    ):
        super().__init__()
        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=1,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=None
        )
        self._device = device
    
    def forward(self, x):
        hidden = None
        return self.net(x), hidden

class PolicyModel(nn.Module):
    def __init__(self, env, cfg, device) -> None:
        self.cfg = cfg
        self.device = device
        super().__init__()

        feature_net_kwargs = cfg.feature_net_kwargs
        feature_net = {}

        for k, v in feature_net_kwargs.items():
            v = dict(v)
            cls = v.pop("cls")
            cls = getattr(F, cls)
            feature_net[k] = cls(**v, device=self.device)

        feature_fusion_kwargs = cfg.feature_fusion
        self.network_model = SimpleFeatureFusion(feature_net, **feature_fusion_kwargs, device=self.device)
        self.actor = Actor(
            input_dim = self.network_model.output_dim,
            action_dim = list(env.single_action_space.nvec),
            device = self.device,
            **cfg.actor,             
        )
        self.critic = Critic(
            input_dim = self.network_model.output_dim,
            device = self.device,
            **cfg.critic,
        )
    
    def get_action_and_value(self, batch, get_action=True, action=None):
        hidden, _ = self.network_model(batch)
        value, _ = self.critic(hidden)

        if not get_action:
            return value
        
        logits, _ = self.actor(hidden)
        dist = self.actor.dist_fn(*logits) if isinstance(logits, tuple) else self.actor.dist_fn(logits)
        if action is None:
            action = dist.mode() if self.actor._deterministic_eval and not self.training else dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, logprob, entropy, value

    
class VisionModel(nn.Module):
    def __init__(self, env, cfg, device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.network_model = build_vision_model(cfg, device)
    
    def get_features(self, images: torch.Tensor):
        BOX_TRESHOLD = 0.35
        TEXT_TRESHOLD = 0.25
        TEXT_PROMPT = self.cfg.agent.prompt

        logits = predict(
            model=self.network_model,
            images=images.cpu().numpy(),
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=self.device,
        )
        return logits