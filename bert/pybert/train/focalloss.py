import torch
import torch.nn as nn


import torch
import torch.nn.functional as F



def focal_loss_multilabel(targets, outputs, gamma=2.0,weight = None, reduction='mean'):
    """
    Focal Loss for multi-label classification.
    
    Args:
    - targets (torch.Tensor): Ground truth labels (binary tensor).
    - outputs (torch.Tensor): Predicted logits.
    - alpha (float): Focal loss hyperparameter, controls the weight assigned to positive class.
    - gamma (float): Focal loss hyperparameter, controls the focus on hard-to-classify examples.
    - reduction (str): Specifies the reduction to apply to the output ('mean', 'sum', or 'none').

    Returns:
    - torch.Tensor: Focal loss value.
    """
    weight = torch.tensor(weight)
    weight = weight.to(device=outputs.device)
    # print(weight.device)
    log_probs = F.logsigmoid(outputs)

    focal_term =  targets * weight *(1 - torch.exp(log_probs))**gamma * -log_probs
    # print("(1- targets) * weight\n",(1- targets) * weight)
    non_focal_term =  (1 - targets) * weight * torch.exp(log_probs)**gamma * -(1 - torch.sigmoid(outputs)).log()
    loss = focal_term + non_focal_term
    # print(loss)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError("Invalid reduction option. Use 'mean', 'sum', or 'none'.")


# if __name__ == '__main__':
#     target = torch.tensor([[0, 1], [1, 1]])
#     input = torch.tensor([[0.1, 0.2], [0.6, 0.7]], requires_grad=True)
#     print(torch.sigmoid(input))
#     print(torch.sigmoid(input).log())
#     output = focal_loss_multilabel(targets=target,outputs = input,weight = [0.1, 0.9])
#     print(torch.nn.BCEWithLogitsLoss()(input=input,target=target.float()))
#     print(output)
#     output.backward()
