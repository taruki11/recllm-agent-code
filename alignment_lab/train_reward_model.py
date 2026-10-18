# -*- coding: utf-8 -*-
"""Pair-wise Reward Model (RM) training step implementation in PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PairwiseRewardLoss(nn.Module):
    """BCE-based Pairwise Reward Model loss formulation.
    Forces the reward of preferred chosen outputs to exceed rejected outputs:
    Loss = -E[log sigmoid(r(x, y_w) - r(x, y_l))]"""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
        # chosen_rewards: [batch_size] (scalar rewards of preferred completions)
        # rejected_rewards: [batch_size] (scalar rewards of rejected completions)
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        return loss

class SimpleRewardModel(nn.Module):
    """Simple linear head over Transformer pooling layer to output scalar rewards."""
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.value_head = nn.Linear(d_model, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: [batch, seq_len, d_model]
        # attention_mask: [batch, seq_len]
        
        # Mean pooling over active tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
        pooled = sum_embeddings / sum_mask # [batch, d_model]
        
        # Calculate scalar rewards
        rewards = self.value_head(pooled).squeeze(-1) # [batch]
        return rewards

def main():
    print("=== PAIRWISE REWARD MODEL SIMULATOR ===")
    
    batch_size, seq_len, d_model = 4, 15, 128
    
    # Mock hidden states from transformer backbones
    torch.manual_seed(42)
    chosen_hidden = torch.randn(batch_size, seq_len, d_model)
    rejected_hidden = torch.randn(batch_size, seq_len, d_model)
    
    # Mask indicating non-padded tokens
    chosen_mask = torch.ones(batch_size, seq_len)
    rejected_mask = torch.ones(batch_size, seq_len)
    chosen_mask[:, 12:] = 0
    rejected_mask[:, 10:] = 0
    
    model = SimpleRewardModel(d_model=d_model)
    loss_fn = PairwiseRewardLoss()
    
    chosen_rewards = model(chosen_hidden, chosen_mask)
    rejected_rewards = model(rejected_hidden, rejected_mask)
    
    loss = loss_fn(chosen_rewards, rejected_rewards)
    
    print(f"Calculated Mock Reward loss: {loss.item():.6f}")
    print(f"  Chosen rewards:   {chosen_rewards.tolist()}")
    print(f"  Rejected rewards: {rejected_rewards.tolist()}")
    print(f"  Margin average:   {(chosen_rewards - rejected_rewards).mean().item():.4f}")

if __name__ == "__main__":
    main()
