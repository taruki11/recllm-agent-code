# -*- coding: utf-8 -*-
"""Direct Preference Optimization (DPO) Implicit Reward training step implementation in PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DPOLoss(nn.Module):
    """Direct Preference Optimization loss computation block.
    Optimizes a policy network directly using preferred (chosen) vs rejected pairs."""
    def __init__(self, beta: float = 0.1, reference_free: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.reference_free = reference_free

    def get_batch_logps(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        average_log_prob: bool = False
    ) -> torch.Tensor:
        """Computes the token-level log probabilities of target labels under logits."""
        # logits shape: [batch, seq_len, vocab_size]
        # labels shape: [batch, seq_len]
        
        # Shift logits and labels to match next token prediction
        action_logits = logits[:, :-1, :]
        action_labels = labels[:, 1:]
        
        # Ignore padding / negative label indices (like -100 in PyTorch)
        loss_mask = action_labels != -100
        
        # Calculate log softmax
        log_probs = F.log_softmax(action_logits, dim=-1)
        
        # Gather the log probabilities of the target labels
        gathered_log_probs = torch.gather(log_probs, dim=-1, index=action_labels.unsqueeze(-1).clamp(min=0)).squeeze(-1)
        
        if average_log_prob:
            return (gathered_log_probs * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
        else:
            return (gathered_log_probs * loss_mask).sum(-1)

    def forward(
        self,
        policy_chosen_logits: torch.Tensor,
        policy_rejected_logits: torch.Tensor,
        reference_chosen_logits: torch.Tensor,
        reference_rejected_logits: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes implicit rewards and DPO loss."""
        # Calculate policy log probabilities
        policy_chosen_logps = self.get_batch_logps(policy_chosen_logits, chosen_labels)
        policy_rejected_logps = self.get_batch_logps(policy_rejected_logits, rejected_labels)
        
        # Calculate reference log probabilities
        if self.reference_free:
            reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
            reference_rejected_logps = torch.zeros_like(policy_rejected_logps)
        else:
            reference_chosen_logps = self.get_batch_logps(reference_chosen_logits, chosen_labels)
            reference_rejected_logps = self.get_batch_logps(reference_rejected_logits, rejected_labels)
            
        # Calculate DPO implicit rewards
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        
        # Calculate loss: -log_sigmoid(chosen_reward - rejected_reward)
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        return loss, chosen_rewards, rejected_rewards

def main():
    print("=== DPO IMPLICIT REWARD LOSS SIMULATOR ===")
    
    # Simulating standard shapes: Batch size 2, Seq length 10, Vocab size 100
    batch_size, seq_len, vocab_size = 2, 10, 100
    
    # Mock logits
    torch.manual_seed(42)
    policy_chosen_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    policy_rejected_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    ref_chosen_logits = torch.randn(batch_size, seq_len, vocab_size)
    ref_rejected_logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Mock labels (using -100 as padding mask)
    chosen_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    chosen_labels[:, :3] = -100
    rejected_labels[:, :4] = -100
    
    dpo = DPOLoss(beta=0.1)
    loss, chosen_rewards, rejected_rewards = dpo(
        policy_chosen_logits, policy_rejected_logits,
        ref_chosen_logits, ref_rejected_logits,
        chosen_labels, rejected_labels
    )
    
    print(f"Calculated Mock DPO Loss: {loss.item():.6f}")
    print(f"Chosen rewards:   {chosen_rewards.detach().tolist()}")
    print(f"Rejected rewards: {rejected_rewards.detach().tolist()}")
    print(f"Reward Margins:   {(chosen_rewards - rejected_rewards).detach().tolist()}")

if __name__ == "__main__":
    main()
