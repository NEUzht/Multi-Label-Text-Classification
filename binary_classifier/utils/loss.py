import torch
import torch.nn.functional as F

def BCE_loss(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def BCE_losses(outputs, targets, weights):

    
    weights = torch.tensor(weights, dtype=torch.float32, device=targets.device)

    loss = sum(weight * torch.nn.BCELoss()(output.squeeze(), target) for weight, output, target in zip(weights, outputs, targets.unbind(dim=1)))
    return loss

def BinaryLoss(outputs, targets, weights, index):
    """
    weights 代表正样本的权重
    """
    output = outputs[:, index].unsqueeze(1)
    target = targets[:, index].unsqueeze(1)
    # print(output,"\n",weights)
    weights = torch.tensor(weights, dtype=torch.float32, device=targets.device)
    weight = weights[index]
    # print(weight)
    log_probs = F.logsigmoid(output)
    # print(log_probs)
    postive_loss = - weight * target * log_probs
    nagtive_loss = -(1 - weight) * (1 - target) *  F.logsigmoid(1 - output)
    loss = postive_loss + nagtive_loss
    # print(loss.mean())
    
    return loss.mean()

