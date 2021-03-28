import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from time import time


def calc_entropy(prob):
    # get entropy
    return torch.sum(-prob * torch.log(prob), dim=1)

def calc_tld(label, logit):
    topk_logit, _ = torch.topk(logit, k=2, dim=1)
    top1_logit = topk_logit[:, 0]
    top2_logit = topk_logit[:, 1]
    
    gt_logit = logit[range(len(logit)), label]
    
    tld_list = torch.where(top1_logit == gt_logit,
                           gt_logit-top2_logit,
                           gt_logit-top1_logit)
    return tld_list

def generate_sample_info(teacher, dataset, root, model_name, device):
    if teacher is None:
        return
    
    sample_info_path = './results/' + "/".join(model_name.split("/")[:2])
    sample_info_path = os.path.join(sample_info_path, 'sample_info.json')
    
    if os.path.isfile(sample_info_path):
        return
        
    if dataset == 'cifar10' or dataset == 'cifar100':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        
        transform_list = transforms.Compose([transforms.RandomCrop(32, padding=4), 
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
        
        batch_size = 128

        # Datasets
        if dataset == 'cifar10':
            train_set = datasets.CIFAR10(root, train=True, transform=transform_list, download=True)
        else:
            train_set = datasets.CIFAR100(root, train=True, transform=transform_list, download=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
        
    elif dataset == 'svhn':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        
        transform_list = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)])
        
        batch_size = 128

        # Datasets
        train_set = datasets.SVHN(root, split='train', transform=transform_list, download=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
        
        
    
    elif dataset == 'tiny-imagenet':
        train_dir = os.path.join(root, 'train')
        
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2770, 0.2691, 0.2821]
        
        transform_list = transforms.Compose([transforms.RandomCrop(64, padding=4),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
        
        batch_size = 128
                
        # Datasets
        train_set = datasets.ImageFolder(train_dir, transform=transform_list)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    
    elif dataset == 'imagenet':
        train_dir = os.path.join(root, 'train')
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]     

        transform_list = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
        
        batch_size = 32
                
        # Datasets
        train_set = datasets.ImageFolder(train_dir, transform=transform_list)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.to(device).eval()
        
    if dataset == 'cifar10' or dataset == 'cifar100' or dataset == 'svhn':
        label_lst = []
        entropy_lst = []
        tld_lst = []

        for i, (image, label) in tqdm(enumerate(train_loader)):
            image = image.type(torch.FloatTensor).to(device)
            label = label.type(torch.LongTensor).to(device)

            label_lst += label.tolist()

            logit = teacher(image)
            prob = F.softmax(logit, dim=1)
            entropy_lst += calc_entropy(prob).tolist()
            tld_lst += calc_tld(label, logit).tolist()

        sample_info = {}
        for i, (label, entropy, tld) in enumerate(zip(label_lst, entropy_lst, tld_lst)):
            sample_info[i] = (label, entropy, tld)

        with open(sample_info_path, "w") as json_file:
            json.dump(sample_info, json_file)
                
    elif dataset == 'tiny-imagenet' or dataset == 'imagenet':
        for idx, (image, label) in enumerate(train_loader):
            label_lst = []
            entropy_lst = []
            tld_lst = []
            
#             image = image.type(torch.FloatTensor).to(device)
            label = label.type(torch.LongTensor).to(device)
            
            label_lst += label.tolist()
#             logit = teacher(image)
#             prob = F.softmax(logit, dim=1)
#             entropy_lst += calc_entropy(prob).tolist()
#             tld_lst += calc_tld(label, logit).tolist()
            entropy_lst += torch.zeros_like(label).tolist()
            tld_lst += torch.zeros_like(label).tolist()
            sample_info = {}
            for i, (label, entropy, tld) in enumerate(zip(label_lst, entropy_lst, tld_lst)):
                sample_info[i] = (label, entropy, tld)
            sample_info_epoch_path = sample_info_path.replace('info', 'info{}'.format(idx))
            
            with open(sample_info_epoch_path, "w") as json_file:
                json.dump(sample_info, json_file)
        label_lst = []
        entropy_lst = []
        tld_lst = []
            
        sample_info_path = './results/' + "/".join(model_name.split("/")[:2])
        sample_info_paths = os.listdir(sample_info_path)
        sample_info_paths = [s for s in sample_info_paths if 'json' in s]

        sample_info = {}
        for idx in range(len(sample_info_paths)):
            sample_info_epoch_path = os.path.join(sample_info_path, 'sample_info{}.json'.format(idx))
            with open(sample_info_epoch_path, "r") as json_file:
                sample_info_epoch_json = json.load(json_file)

            for k, v in sample_info_epoch_json.items():
                sample_info[str(batch_size*idx+int(k))] = v

        sample_info_path = os.path.join(sample_info_path, 'sample_info.json')
        with open(sample_info_path, "w") as json_file:
            json.dump(sample_info, json_file)