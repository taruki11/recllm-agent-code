# -*- coding: utf-8 -*-
"""ORPO (Odds Ratio Preference Optimization) training step implementation in PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ORPOLoss(nn.Module):
    """ORPO Loss: Merges SFT CrossEntropy and Odds Ratio preference penalties
    into a single step without needing an expensive Reference Model."""
    def __init__(self, alpha: float = 1.0, pad_token_id: int = -100) -> None:
        super().__init__()
        self.alpha = alpha
        self.pad_token_id = pad_token_id

    def get_batch_logps(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        action_logits = logits[:, :-1, :]
        action_labels = labels[:, 1:]
        
        loss_mask = action_labels != self.pad_token_id
        log_probs = F.log_softmax(action_logits, dim=-1)
        gathered = torch.gather(log_probs, dim=-1, index=action_labels.unsqueeze(-1).clamp(min=0)).squeeze(-1)
        return (gathered * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)

    def forward(
        self,
        policy_chosen_logits: torch.Tensor,
        policy_rejected_logits: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Compute SFT loss on chosen output (standard teacher forcing)
        # Shift logits and labels
        action_logits = policy_chosen_logits[:, :-1, :].contiguous()
        action_labels = chosen_labels[:, 1:].contiguous()
        
        sft_loss = F.cross_entropy(
            action_logits.view(-1, action_logits.shape[-1]), 
            action_labels.view(-1), 
            ignore_index=self.pad_token_id
        )
        
        # 2. Compute Odds Ratio Log probabilities
        chosen_logps = self.get_batch_logps(policy_chosen_logits, chosen_labels)
        rejected_logps = self.get_batch_logps(policy_rejected_logits, rejected_labels)
        
        # Log odds formulation: log (P / (1 - P)) = log P - log(1 - P)
        # We approximate P as exp(log_prob)
        chosen_odds = chosen_logps - torch.log1p(-torch.exp(chosen_logps).clamp(max=0.99))
        rejected_odds = rejected_logps - torch.log1p(-torch.exp(rejected_logps).clamp(max=0.99))
        
        # Odds ratio loss: -log_sigmoid(chosen_odds - rejected_odds)
        or_loss = -F.logsigmoid(chosen_odds - rejected_odds).mean()
        
        # Combined loss
        total_loss = sft_loss + self.alpha * or_loss
        
        return total_loss, sft_loss, or_loss

def main():
    print("=== ORPO LOSS SIMULATOR ===")
    
    batch_size, seq_len, vocab_size = 2, 10, 100
    
    torch.manual_seed(42)
    policy_chosen_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    policy_rejected_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    
    chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    chosen_labels[:, :3] = -100
    rejected_labels[:, :4] = -100
    
    orpo = ORPOLoss(alpha=1.0)
    total_loss, sft_loss, or_loss = orpo(
        policy_chosen_logits, policy_rejected_logits,
        chosen_labels, rejected_labels
    )
    
    print(f"Calculated Mock ORPO Total Loss: {total_loss.item():.6f}")
    print(f"  SFT CrossEntropy Component:  {sft_loss.item():.6f}")
    print(f"  Odds Ratio Penalty Component: {or_loss.item():.6f}")

if __name__ == "__main__":
    main()
