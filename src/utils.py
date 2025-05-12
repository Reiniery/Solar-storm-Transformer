import torch
import torch.nn as nn
import os
from src.dataset import Multimodal_Datasets


def get_data(args, dataset, split='train'):
    data_path = os.path.join(args.data_path, dataset) + f'_{split}.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, dataset, split)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path, weights_only=False)
    return data


def save_load_name(args, name=''):
    load_name = name + '_' + args.model
    return load_name


def save_model(args, model, name=''):
    if not os.path.exists('output/'):
        os.makedirs('output/')
    name = save_load_name(args, name)
    torch.save(model, f'output/{args.name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'output/{args.name}.pt', weights_only=False)
    return model


########## remake_label v1 - reclass -1 values to positive
# def remake_label(target, num_classes=3):
#     """
#     Remap labels to ensure they are valid for the model's output.

#     Args:
#         target (torch.Tensor): The target tensor containing labels.
#         num_classes (int): The number of classes.

#     Returns:
#         torch.Tensor: Remapped labels.
#     """
#     # Ensure target values are in the range [0, num_classes - 1]
#     if (target < 0).any() or (target >= 3).any():
#         print("Invalid target values detected! Remapping to valid range.")
#         target = torch.clamp(target, min=0, max=3)
    
#     return target

########## remake _label v2- change dimensions
def remake_label(target, num_classes=4):
    # print("REMAKE_LABEL: DATA DIMENSION: ",target.dim())
    # print("REMAKE_LABEL: DATA DIMENSION: ",target.size())
    # print("Max/min", target.max(), target.min())
    # print("Unique values:", torch.unique(target))
    if (target < 0).any() or (target >= 4).any():
        # print("Invalid target values detected! Remapping to valid range.")
        target = torch.clamp(target, min=0, max=target.max()+1)
    if target.dim()>2:
        target= target.view(input.size(0),input.size(1),-1) 
        target= target.transpose(1,2)
        target =target.contiguous().view(-1,input.size(2))
        
    return target.view (-1,1)

class focalloss(nn.Module):
    def __init__(self, alpha=[0.1, 0.1, 0.8], gamma=2, reduction='mean'):
        super(focalloss, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        print("PRED SHAPE", pred.size())
        print("target SHAPE", target.size())
        target = remake_label(target).type(torch.int64)
        alpha = self.alpha[target]
        log_softmax = torch.log_softmax(pred, dim=0)
        logpt = torch.gather(log_softmax, dim=0, index=target.view(-1, 1))
        logpt = logpt.view(-1)
        ce_loss = -logpt 
        pt = torch.exp(logpt)
        focal_loss = alpha * ((1 - pt) ** self.gamma * ce_loss).t()
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss
