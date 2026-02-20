import torch
import torch.nn.functional as F
import sys
import os

# Add src to path to import our changes
sys.path.append("/scratch/gpfs/ZHUANGL/jl0796/LLaMA-Factory-AutoReviewer/src")
from llamafactory.train.trainer_utils import weighted_bce_loss_accept

def test_weighted_loss():
    print("Testing weighted_bce_loss_accept...")
    
    # Mock token IDs
    accept_ids = [16646]
    reject_ids = [78413]
    gamma = 2.0
    
    # Create mock logits (N, Vocab)
    # N = batch_size * seq_len. Let's say batch_size=2, seq_len=4.
    vocab_size = 100000
    logits = torch.zeros((8, vocab_size))
    
    # Create mock labels (batch_size, seq_len)
    # Sample 0: Accept at index 2
    # Sample 1: Reject at index 1
    labels = torch.full((2, 4), -100, dtype=torch.long)
    labels[0, 2] = 16646
    labels[1, 1] = 78413
    
    # Set specific logits for Accept/Reject
    # Sample 0: L_A=2.0, L_R=1.0 -> logit_diff = 1.0 (Expect p=sig(1.0)=0.731)
    # Decision position for Sample 0 is index 2, so it's row 2 in flattened logits (Wait, shifted labels? No, labels[0,2] corresponds to logits[0,1] predict? No, wait.)
    # In weighted_bce_loss_accept:
    # labels = pad(labels, (0,1)) -> shift_labels = labels[..., 1:] -> logits correspond to shift_labels
    # So if shift_labels[0,1] is the decision token, it uses logits[0,1] to predict it.
    
    # Let's align our mock:
    # labels: [[-100, -100, 16646, -100], [-100, 78413, -100, -100]]
    # padded labels: [[-100, -100, 16646, -100, -100], [-100, 78413, -100, -100, -100]]
    # shift_labels: [[-100, 16646, -100, -100], [78413, -100, -100, -100]] (length 4)
    # logits: (batch_size * 4, vocab_size) = (8, vocab_size)
    # Decision positions in flattened shift_labels:
    # sample 0, index 1 -> position 1
    # sample 1, index 0 -> position 4
    
    # Row 1 (Sample 0 decision): L_A=2.0, L_R=1.0
    logits[1, 16646] = 2.0
    logits[1, 78413] = 1.0
    
    # Row 4 (Sample 1 decision): L_A=1.0, L_R=3.0
    logits[4, 16646] = 1.0
    logits[4, 78413] = 3.0
    
    outputs = {"logits": logits.view(2, 4, vocab_size)}
    
    # Hand calculation:
    # Sample 0 (Accept): target_y = 1.0, noisy_y = (1-1)*2 + 1 = 1.0. p = sig(2.0-1.0) = sig(1.0)
    # Sample 1 (Reject): target_y = 0.0, noisy_y = (1-0)*2 + 0 = 2.0. p = sig(1.0-3.0) = sig(-2.0)
    # Loss = (BCE(sig(1.0), 1.0) + BCE(sig(-2.0), 2.0)) / 2
    
    p0 = torch.sigmoid(torch.tensor(1.0))
    p1 = torch.sigmoid(torch.tensor(-2.0))
    expected_loss0 = F.binary_cross_entropy(p0, torch.tensor(1.0))
    expected_loss1 = F.binary_cross_entropy(p1, torch.tensor(2.0))
    expected_avg_loss = (expected_loss0 + expected_loss1) / 2.0
    
    actual_loss = weighted_bce_loss_accept(outputs, labels, gamma=gamma, accept_ids=accept_ids, reject_ids=reject_ids)
    
    print(f"Expected Loss (p0={p0:.4f}, p1={p1:.4f}): {expected_avg_loss:.4f}")
    print(f"Actual Loss: {actual_loss:.4f}")
    
    if torch.allclose(actual_loss, expected_avg_loss):
        print("Success! The loss calculation matches.")
    else:
        print("Failure! Loss mismatch.")

if __name__ == "__main__":
    test_weighted_loss()
