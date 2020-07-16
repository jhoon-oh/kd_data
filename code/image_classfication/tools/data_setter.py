import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import datasets
import torchvision.transforms as transforms
import random
import numpy as np
import os
import json

from time import time
from tqdm import tqdm
from transforms.autoaugment import CIFAR10Policy, ImageNetPolicy
from transforms.cutout import *

__all__ = ['cifar_10_setter', 'cifar_100_setter', 'tiny_imagenet_setter', 'imagenet_setter']

def set_train_valid(dataset, root, teacher, train_set, model_name, cls_acq, cls_order, delta, sample_acq, sample_order, zeta):
    # cls_acq = 'random' or 'entropy'
    # cls_order = 'highest' or 'lowest' (when the acquisition fucntion for classes is entropy)
    # delta = the ratio of train classes
    # sample_acq = 'random' or 'entropy'
    # sample_order = 'highest' or 'lowest' (when the acquisition fucntion for samples is entropy)
    # zeta = the ratio of train samples per train class
    # e.g., delta=1.0, zeta=1.0 -> train classes # = 100 & train samples # per train class = 500
    # e.g., delta=0.9, zeta=0.0 -> train classes # = 90 & train samples # per train class = 500
    # e.g., delta=0.0, zeta=0.9 -> train classes # = 100 & train samples # per train class = 450
    # e.g., delta=0.9, zeta=0.9 -> train classes # = 90 & train samples # per train class = 450
    
    if teacher is None:
        label_lst = []
        
        if dataset == 'imagenet':
            for cls_name in train_set.classes:
                cls_path = os.path.join(root, 'train', cls_name)
                cls_lst = [f for f in os.listdir(cls_path) if cls_name in f]
                cls_id = int(train_set.class_to_idx[cls_name])
                label_lst += [cls_id] * len(cls_lst)
        
        else:
            for _, label in train_set:
                label_lst.append(label)
        
        label_lst = np.array(label_lst)

        dic = {}
        for cls_id in sorted(np.unique(label_lst)):
            dic[str(cls_id)] = np.where(label_lst==cls_id)[0]
            
        train_dict = {}
        valid_dict = {}

        total_cls_number = len(dic.keys())
        valid_cls_number = int(total_cls_number * (1-delta))

        valid_cls_id = sorted(random.sample(dic.keys(), valid_cls_number))
        train_cls_id = sorted(list(set(dic.keys())-set(valid_cls_id)))

        for cls_id in valid_cls_id:
            train_dict[cls_id] = []
            valid_dict[cls_id] = [int(_) for _ in dic[cls_id]]

        for cls_id in train_cls_id:
            sample_lst = dic[cls_id]

            total_cls_sample_number = len(sample_lst)
            valid_cls_sample_number = int(total_cls_sample_number * (1-zeta))

            valid_dict[cls_id] = [int(_) for _ in sorted(random.sample(list(sample_lst), valid_cls_sample_number))]
            train_dict[cls_id] = [int(_) for _ in sorted(list(set(sample_lst)-set(valid_dict[cls_id])))]

        train_json_path = './results/' + model_name +'_train.json'
        valid_json_path = './results/' + model_name +'_valid.json'
        with open(train_json_path, "w") as json_file:
            json.dump(train_dict, json_file)
        with open(valid_json_path, "w") as json_file:
            json.dump(valid_dict, json_file)

        train_list = []
        valid_list = []
        for _, v in train_dict.items():
            train_list += v
        for _, v in valid_dict.items():
            valid_list += v
            
    else:
        sample_info_path = './results/' + "/".join(model_name.split("/")[:2])
        sample_info_path = os.path.join(sample_info_path, 'sample_info.json')
        with open(sample_info_path, "r") as json_file:
            sample_info = json.load(json_file)

        label_lst = []
        entropy_lst = []
        for _, (label, entropy) in sample_info.items():
            label_lst.append(label)
            entropy_lst.append(entropy)
        label_lst = np.array(label_lst)
        entropy_lst = np.array(entropy_lst)

        dic = {}
        for cls_id in sorted(np.unique(label_lst)):
            dic[str(cls_id)] = np.where(label_lst==cls_id)[0]

        train_dict = {}
        valid_dict = {}

        total_cls_number = len(dic.keys())
        valid_cls_number = int(total_cls_number * (1-delta))

        if cls_acq == 'random':
            valid_cls_id = sorted(random.sample(dic.keys(), valid_cls_number))
            train_cls_id = sorted(list(set(dic.keys())-set(valid_cls_id)))

        elif cls_acq == 'entropy':
            cls_entropy_lst = []
            for _, v in dic.items():
                cls_entropy_lst.append(np.mean(entropy_lst[v]))
            cls_entropy_lst = np.array(cls_entropy_lst)

            if cls_order == 'highest':
                valid_cls_id = list((-cls_entropy_lst).argsort()[:valid_cls_number])
            elif cls_order == 'lowest':
                valid_cls_id = list((cls_entropy_lst).argsort()[:valid_cls_number])
            valid_cls_id = [str(_) for _ in valid_cls_id]
            train_cls_id = sorted(list(set(dic.keys())-set(valid_cls_id)))

        for cls_id in valid_cls_id:
            train_dict[cls_id] = []
            valid_dict[cls_id] = [int(_) for _ in dic[cls_id]]

        for cls_id in train_cls_id:
            sample_lst = dic[cls_id]

            total_cls_sample_number = len(sample_lst)
            valid_cls_sample_number = int(total_cls_sample_number * (1-zeta))

            if sample_acq == 'random':
                valid_dict[cls_id] = [int(_) for _ in sorted(random.sample(list(sample_lst), valid_cls_sample_number))]
                train_dict[cls_id] = [int(_) for _ in sorted(list(set(sample_lst)-set(valid_dict[cls_id])))]

            elif sample_acq == 'entropy':
                sample_entropy_lst = entropy_lst[sample_lst]

                if sample_order == 'highest':
                    valid_sample_id = list((-sample_entropy_lst).argsort()[:valid_cls_sample_number])
                elif sample_order == 'lowest':
                    valid_sample_id = list((sample_entropy_lst).argsort()[:valid_cls_sample_number])
                valid_dict[cls_id] = [int(_) for _ in sorted(list(np.array(sample_lst)[valid_sample_id]))]
                train_dict[cls_id] = [int(_) for _ in sorted(list(set(sample_lst)-set(valid_dict[cls_id])))]

        train_json_path = './results/' + model_name +'_train.json'
        valid_json_path = './results/' + model_name +'_valid.json'
        with open(train_json_path, "w") as json_file:
            json.dump(train_dict, json_file)
        with open(valid_json_path, "w") as json_file:
            json.dump(valid_dict, json_file)

        train_list = []
        valid_list = []
        for _, v in train_dict.items():
            train_list += v
        for _, v in valid_dict.items():
            valid_list += v
    
    return train_list, valid_list

def train_transform(mean, std, data, mode):
    '''
    mode: vanilla, flip, crop, auto, fastauto
    data: cifar10, cifar100, imagenet
    '''
    train_transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)]
    
    if data == 'cifar10' or data == 'cifar100':
        if mode == 'flip':
            train_transform_list = [transforms.RandomHorizontalFlip()] + train_transform_list
        elif mode == 'crop':
            train_transform_list = [transforms.RandomCrop(32, padding=4), 
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)]

        elif mode == 'auto':
            train_transform_list = [transforms.RandomCrop(32, padding=4, fill = 128),
                                    transforms.RandomHorizontalFlip(),CIFAR10Policy(),
                                    transforms.ToTensor(),
                                    Cutout(n_holes=1, length=16),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465) , (0.2470, 0.243, 0.261))]
    elif data == 'tiny-imagenet':
        train_transform_list = [transforms.RandomCrop(64, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2770, 0.2691, 0.2821])]
    elif data == 'imagenet':
        if mode == 'flip':
            train_transform_list = [transforms.RandomHorizontalFlip()] + train_transform_list
        elif mode == 'crop':
            train_transform_list = [transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)]
    
    return train_transform_list


def cifar_10_setter(teacher,
                    mode,
                    batch_size,
                    root,
                    model_name,
                    cls_acq,
                    cls_order,
                    delta,
                    sample_acq,
                    sample_order,
                    zeta,
                    pin_memory=False,
                    num_workers=4,
                    download=True,
                    fixed_valid=True):
    if fixed_valid:
        random.seed(2020)
    # Data augmentation and normalization for training
    # Just normalization for validation
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    train_transform_list = train_transform(mean, std, data='cifar10', mode=mode)

    train_transforms = transforms.Compose(train_transform_list)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    batch_size = batch_size
    
    # Datasets
    train_set = datasets.CIFAR10(root, train=True, transform=train_transforms, download=download) # train transform applied
    train_list, valid_list = set_train_valid(dataset='cifar10',
                                             root=root,
                                             teacher=teacher,
                                             train_set=train_set,
                                             model_name=model_name,
                                             cls_acq=cls_acq,
                                             cls_order=cls_order,
                                             delta=delta,
                                             sample_acq=sample_acq,
                                             sample_order=sample_order,
                                             zeta=zeta)
    cifar10_train_set = Subset(train_set, train_list)
    cifar10_valid_set = Subset(train_set, valid_list)
    
    cifar10_test_set = datasets.CIFAR10(root, train=False, transform=test_transforms, download=download)

    train_loader = torch.utils.data.DataLoader(cifar10_train_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(cifar10_valid_set, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(cifar10_test_set, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)

    dataloaders = {'train': train_loader,
                   'valid': valid_loader,
                   'test': test_loader,}

    dataset_sizes = {'train': len(cifar10_train_set),
                     'valid': len(cifar10_valid_set),
                     'test': len(cifar10_test_set)}
    
    return dataloaders, dataset_sizes

def cifar_100_setter(teacher,
                     mode,
                     batch_size,
                     root,
                     model_name,
                     cls_acq,
                     cls_order,
                     delta,
                     sample_acq,
                     sample_order,
                     zeta,
                     pin_memory=False,
                     num_workers=4,
                     download=True,
                     fixed_valid=True):
    if fixed_valid:
        random.seed(2020)
    # Data augmentation and normalization for training
    # Just normalization for validation
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2762)
    train_transform_list = train_transform(mean, std, data='cifar100', mode=mode)
    
    train_transforms = transforms.Compose(train_transform_list)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    batch_size = batch_size
    
    # Datasets
    train_set = datasets.CIFAR100(root, train=True, transform=train_transforms, download=download)
    train_list, valid_list = set_train_valid(dataset='cifar100',
                                             root=root,
                                             teacher=teacher,
                                             train_set=train_set,
                                             model_name=model_name,
                                             cls_acq=cls_acq,
                                             cls_order=cls_order,
                                             delta=delta,
                                             sample_acq=sample_acq,
                                             sample_order=sample_order,
                                             zeta=zeta)
    cifar100_train_set = Subset(train_set, train_list)
    cifar100_valid_set = Subset(train_set, valid_list)
    
    cifar100_test_set = datasets.CIFAR100(root, train=False, transform=test_transforms, download=download)

    train_loader = torch.utils.data.DataLoader(cifar100_train_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(cifar100_valid_set, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(cifar100_test_set, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)

    dataloaders = {'train': train_loader,
                   'valid': valid_loader,
                   'test': test_loader,}

    dataset_sizes = {'train': len(cifar100_train_set),
                     'valid': len(cifar100_valid_set),
                     'test': len(cifar100_test_set)}
    
    return dataloaders, dataset_sizes

def tiny_imagenet_setter(teacher,
                    mode,
                    batch_size,
                    model_name,
                    cls_acq,
                    cls_order,
                    delta,
                    sample_acq,
                    sample_order,
                    zeta,
                    root='/home/taehyeon/tiny-imagenet/',
                    pin_memory=False,
                    num_workers=4,
                    download=True,
                    fixed_valid=True):
    if fixed_valid:
        random.seed(2020)
    train_dir = os.path.join(root, 'train')
    test_dir = os.path.join(root, 'val')
        
    mean = [0.4802, 0.4481, 0.3975]
    std = [0.2770, 0.2691, 0.2821]  
    
    train_transform_list = train_transform(mean, std, data='tiny-imagenet', mode=mode)

    train_transforms = transforms.Compose(train_transform_list)
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    batch_size = batch_size
    
    
    # Datasets
    train_set = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_list, valid_list = set_train_valid(dataset='tiny-imagenet',
                                             root=root,
                                             teacher=teacher,
                                             train_set=train_set,
                                             model_name=model_name,
                                             cls_acq=cls_acq,
                                             cls_order=cls_order,
                                             delta=delta,
                                             sample_acq=sample_acq,
                                             sample_order=sample_order,
                                             zeta=zeta)
    tiny_imagenet_train_set = Subset(train_set, train_list)
    tiny_imagenet_valid_set = Subset(train_set, valid_list)
    tiny_imagenet_test_set = datasets.ImageFolder(test_dir, transform=test_transforms)

    pin_memory = True
    train_loader = torch.utils.data.DataLoader(tiny_imagenet_train_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(tiny_imagenet_valid_set, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(tiny_imagenet_test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

    dataloaders = {'train' : train_loader,
                   'valid' : valid_loader,
                   'test' : test_loader,}

    dataset_sizes = {'train': len(tiny_imagenet_train_set),
                     'valid': len(tiny_imagenet_valid_set),
                     'test': len(tiny_imagenet_test_set)}  
    
    return dataloaders, dataset_sizes

def imagenet_setter(teacher,
                    mode,
                    batch_size,
                    model_name,
                    cls_acq,
                    cls_order,
                    delta,
                    sample_acq,
                    sample_order,
                    zeta,
                    root='/home/taehyeon/ImageNet/Data/',
                    pin_memory=False,
                    num_workers=4,
                    download=True,
                    fixed_valid=True):
    if fixed_valid:
        random.seed(2020)
    train_dir = os.path.join(root, 'train')
    test_dir = os.path.join(root, 'val')
        
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]     
    
    train_transform_list = train_transform(mean, std, data='imagenet', mode=mode)

    train_transforms = transforms.Compose(train_transform_list)
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    batch_size = batch_size
    
    
    # Datasets
    train_set = datasets.ImageFolder(train_dir, transform=train_transforms)
    train_list, valid_list = set_train_valid(dataset='imagenet',
                                             root=root,
                                             teacher=teacher,
                                             train_set=train_set,
                                             model_name=model_name,
                                             cls_acq=cls_acq,
                                             cls_order=cls_order,
                                             delta=delta,
                                             sample_acq=sample_acq,
                                             sample_order=sample_order,
                                             zeta=zeta)
    imagenet_train_set = Subset(train_set, train_list)
    imagenet_valid_set = Subset(train_set, valid_list)
    imagenet_test_set = datasets.ImageFolder(test_dir, transform=test_transforms)

    pin_memory = True
    train_loader = torch.utils.data.DataLoader(imagenet_train_set, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(imagenet_valid_set, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(imagenet_test_set, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

    dataloaders = {'train': train_loader,
                   'valid': valid_loader,
                   'test': test_loader,}

    dataset_sizes = {'train': len(imagenet_train_set),
                     'valid': len(imagenet_valid_set),
                     'test': len(imagenet_test_set)}  
    
    return dataloaders, dataset_sizes
