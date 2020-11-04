import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['cr_loss', 'mse_logit', 'smooth_logit', 'kd_loss', 'diri_loss', 'inf_grad', 'ze_grad']

def cr_loss(output, label):
    """
    from:
        https://github.com/peterliht/knowledge-distillation-pytorch
    
    Compute the cross entropy loss given outputs and labels.
    Args:
        outputs: (Variable) dimension batch_size x 6 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    return nn.CrossEntropyLoss()(output, label)

def mse_logit(output, teacher_output):
    
    return nn.MSELoss()(output, teacher_output)

def smooth_logit(output, teacher_output):
    
    return nn.SmoothL1Loss()(output, teacher_output)


def kd_loss(output, label, teacher_output, alpha, temperature):
    """
    from:
        https://github.com/peterliht/knowledge-distillation-pytorch
        
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = alpha
    T = temperature
    
    """
    kd_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(output/T, dim=1),
                                             F.softmax(teacher_output/T, dim=1)).type(torch.FloatTensor).cuda(gpu)
    
    kd_loss = kd_filter * torch.sum(kd_loss, dim=1) # kd filter is filled with 0 and 1.
    kd_loss = torch.sum(kd_loss) / torch.sum(kd_filter) * (alpha * T * T)
    """
    
    kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output/T, dim=1),
                                                  F.softmax(teacher_output/T, dim=1)) * (alpha * T * T)

    cr_loss = nn.CrossEntropyLoss()(output, label) * (1. - alpha)
    
    return kd_loss + cr_loss

class DiriAdaptiveLabelLoss(nn.Module):
    def __init__(self, device, confusion, smoothing=0.0, dim=-1):
        super(DiriAdaptiveLabelLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim
        self.confusion = confusion
        self.device = device

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred).to(self.device)
            for i in range(target.shape[0]):
                diri = torch.distributions.dirichlet.Dirichlet(torch.tensor(self.confusion[target[i]]))
                tmp = diri.sample()
                true_dist[i][:target[i]] = tmp[:target[i]] * self.smoothing
                true_dist[i][target[i]] = self.confidence
                true_dist[i][target[i]+1:] = tmp[target[i]:] * self.smoothing
                
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    
def diri_loss(output, label, alpha, confusion, gpu):
    """
    diri loss
    """
    diri_loss = DiriAdaptiveLabelLoss(device=gpu, confusion=confusion, smoothing=alpha)(pred=output, target=label)
        
    return diri_loss

def inf_grad(pred, target, classes, gpu):
    custom_grad = classes * (pred-target) - torch.sum(pred-target, dim=1, keepdim=True)
    #print(torch.sum(pred-target, dim=1, keepdim=True).shape)

    return custom_grad/(classes*classes) /pred.shape[0]

def ze_grad(pred, target, classes, gpu):
    pred_grad = torch.zeros_like(pred).to(gpu)
    target_grad = torch.zeros_like(target).to(gpu)
    pred_argm = torch.argmax(pred, 1)
    target_argm = torch.argmax(target, 1)
    for i in range(target.shape[0]):
        pred_grad[i, pred_argm[i]] += 1.0
        target_grad[i, target_argm[i]] += 1.0
    
    return pred_grad-target_grad
