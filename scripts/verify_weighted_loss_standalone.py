import torch
import torch.nn.functional as F

def _get_binary_decision_logit(
    logits: torch.Tensor,
    labels: torch.Tensor,
    accept_ids: list[int],
    reject_ids: list[int],
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = labels != ignore_index
    if not valid_mask.any():
        return torch.tensor([], device=logits.device), torch.tensor([], device=logits.device)

    accept_ids_tensor = torch.tensor(accept_ids, device=labels.device)
    reject_ids_tensor = torch.tensor(reject_ids, device=labels.device)

    is_accept = torch.any(labels.unsqueeze(-1) == accept_ids_tensor, dim=-1)
    is_reject = torch.any(labels.unsqueeze(-1) == reject_ids_tensor, dim=-1)
    is_decision = is_accept | is_reject

    if not is_decision.any():
        return torch.tensor([], device=logits.device), torch.tensor([], device=logits.device)

    decision_logits = logits[is_decision]
    decision_labels = labels[is_decision]

    l_accept = torch.max(decision_logits[:, accept_ids], dim=-1)[0]
    l_reject = torch.max(decision_logits[:, reject_ids], dim=-1)[0]

    logit_diff = l_accept - l_reject
    target_y = torch.any(decision_labels.unsqueeze(-1) == accept_ids_tensor, dim=-1).float()

    return logit_diff, target_y


def weighted_bce_loss_accept(outputs, labels, gamma=1.0, accept_ids=None, reject_ids=None, num_items_in_batch=None):
    logits = outputs.get("logits")
    if logits is None:
        return outputs.get("loss", torch.tensor(0.0))

    if accept_ids is None or reject_ids is None:
        accept_ids = [16646]
        reject_ids = [78413]

    vocab_size = logits.size(-1)
    labels_padded = torch.nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels_padded[..., 1:].contiguous()
    logits_flat = logits.view(-1, vocab_size)
    shift_labels_flat = shift_labels.view(-1).to(logits.device)

    logit_diff, target_y = _get_binary_decision_logit(logits_flat, shift_labels_flat, accept_ids, reject_ids)

    if logit_diff.numel() == 0:
        return torch.nn.functional.cross_entropy(logits_flat, shift_labels_flat, ignore_index=-100)

    p_star = torch.sigmoid(logit_diff)
    weight = (1.0 - target_y) * gamma + target_y

    return torch.nn.functional.binary_cross_entropy(p_star, target_y, weight=weight.float())

def test_weighted_loss():
    print("Testing weighted_bce_loss_accept standalone...")
    
    accept_ids = [1]
    reject_ids = [2]
    gamma = 2.0
    
    vocab_size = 10
    logits = torch.zeros((2, 4, vocab_size))
    labels = torch.full((2, 4), -100, dtype=torch.long)
    
    # Sample 0: Accept at index 2 (corresponds to shift_label index 1, predicted by logit index 1)
    labels[0, 2] = 1
    logits[0, 1, 1] = 2.0
    logits[0, 1, 2] = 1.0
    
    # Sample 1: Reject at index 1 (corresponds to shift_label index 0, predicted by logit index 0)
    labels[1, 1] = 2
    logits[1, 0, 1] = 1.0
    logits[1, 0, 2] = 3.0
    
    outputs = {"logits": logits}
    
    p0 = torch.sigmoid(torch.tensor(1.0)) # 2.0 - 1.0
    p1 = torch.sigmoid(torch.tensor(-2.0)) # 1.0 - 3.0
    y0 = 1.0
    y1 = 0.0
    weight0 = (1.0 - y0) * gamma + y0 # 1.0
    weight1 = (1.0 - y1) * gamma + y1 # 2.0
    
    expected_loss0 = F.binary_cross_entropy(p0, torch.tensor(y0), weight=torch.tensor(weight0))
    expected_loss1 = F.binary_cross_entropy(p1, torch.tensor(y1), weight=torch.tensor(weight1))
    expected_avg_loss = (expected_loss0 + expected_loss1) / 2.0
    
    actual_loss = weighted_bce_loss_accept(outputs, labels, gamma=gamma, accept_ids=accept_ids, reject_ids=reject_ids)
    
    print(f"Expected Loss: {expected_avg_loss:.4f}")
    print(f"Actual Loss: {actual_loss:.4f}")
    
    assert torch.allclose(actual_loss, expected_avg_loss)
    print("Success!")

if __name__ == "__main__":
    test_weighted_loss()
