"""VLA Diffusion Policy Model."""

from typing import Optional
import torch
import torch.nn as nn
#from .encoders import ImageEncoderTinyCNN, TextEncoderTransformer, StateEncoderMLP
from .encoders import ImageEncoderTinyCNN, TextEncoderTinyGRU, StateEncoderMLP
from .fusion import FusionMLP
from .diffusion_head import DiffusionConfig, DiffusionPolicyHead
from .flow_matching_head import FlowMatchingConfig, FlowMatchingPolicyHead

class VLADiffusionPolicy(nn.Module):
    def __init__(self, vocab_size, state_dim, action_dim,
                 d_model=128, diffusion_T=16, use_flow_matching=False):
        super().__init__()
        self.img_encoder = ImageEncoderTinyCNN(d_model=d_model)
        self.txt_encoder = TextEncoderTinyGRU(vocab_size=vocab_size, d_word=64, d_model=d_model)
        self.state_encoder = StateEncoderMLP(state_dim=state_dim, d_model=d_model)
        self.fusion = FusionMLP(d_model=d_model)

        self.use_flow_matching = use_flow_matching
        if self.use_flow_matching:
            fm_cfg = FlowMatchingConfig(
                action_dim=action_dim,
                cond_dim=d_model)
            self.policy_head = FlowMatchingPolicyHead(fm_cfg)
        else:
            cfg = DiffusionConfig(
            T=diffusion_T,
            action_dim=action_dim,
            cond_dim=d_model,
            )
            self.policy_head = DiffusionPolicyHead(cfg)


    def encode_obs(self, image, text_tokens, state, 
                   gamma: Optional[torch.Tensor] = None, 
                   beta: Optional[torch.Tensor] = None):
        """
        Encode observations and fuse them with optional FiLM modulation.
        
        Args:
            image: (B, 3, H, W) input image
            text_tokens: (B, T_text) tokenized text
            state: (B, state_dim) robot state
            gamma: Optional FiLM gamma parameter
            beta: Optional FiLM beta parameter
        
        Returns:
            fused_context: (B, d_model) fused context embedding
        """
        img_token = self.img_encoder(image)  # (B, d_model)
        txt_token = self.txt_encoder(text_tokens)  # (B, d_model)
        state_token = self.state_encoder(state)  # (B, d_model)
        fused_context = self.fusion(img_token, txt_token, state_token, gamma=gamma, beta=beta)
        return fused_context
    
    def encode_text(self, text_tokens):
        """
        Encode text tokens only for generating gamma/beta from new prompt.
        """
        return self.txt_encoder(text_tokens)

    def loss(self, image, text_tokens, state, actions):
        """
        Compute the loss of the diffusion policy head given the image, text tokens, state, and actions.
        """
        fused_context = self.encode_obs(image, text_tokens, state)
        return self.policy_head.loss(actions, fused_context)

    def act(self, image, text_tokens, state,
            gamma: Optional[torch.Tensor] = None,
            beta: Optional[torch.Tensor] = None):
        """
        Generate actions with optional FiLM modulation (inference only).
        
        Args:
            image: (B, 3, H, W)
            text_tokens: (B, T_text)
            state: (B, state_dim)
            gamma: Optional FiLM gamma parameter from LLM generator
            beta: Optional FiLM beta parameter from LLM generator
            
        Returns:
            actions: (B, action_dim)
        """
        fused_context = self.encode_obs(image, text_tokens, state, gamma=gamma, beta=beta)
        actions = self.policy_head.sample(fused_context)
        return actions