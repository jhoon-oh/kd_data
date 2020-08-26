import torch
from models import cifar, imagenet

from .data_setter import *

def load_model(teacher_str, student_str, dataset, device):
    # teacher_str, student_str: string ex) wrn-28-4, res-28-4
    # nobn: the affine parameters in bn get False
    # demo: no shortcut
    
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'tiny-imagenet':
        num_classes = 200
    elif dataset == 'imagenet':
        num_classes = 1000
    
    if dataset == 'cifar10' or dataset == 'cifar100':
        if teacher_str is not None:
            bn_aff = False if 'nobn' in teacher_str else True
            shortcut = False if 'demo' in teacher_str else True

            if 'wrn' in teacher_str:
                teacher_depth = int(teacher_str.split('-')[1])
                teacher_widen_factor = int(teacher_str.split('-')[2])
                teacher = cifar.WideResNet(depth=teacher_depth, widen_factor=teacher_widen_factor, num_classes=num_classes, bn_aff = bn_aff, shortcut = shortcut)
            elif 'res' in teacher_str:
                teacher_depth = int(teacher_str.split('-')[1])
                teacher_widen_factor = int(teacher_str.split('-')[2])
                teacher = cifar.ResNet(depth=teacher_depth, width=teacher_widen_factor, num_classes=num_classes, bn_aff = bn_aff, shortcut = shortcut)         
                
            filename = './model_checkpoints/{}/None/{}/alp_0.1_T_1.0/random_highest_1.0_random_highest_1.0_seed9999.t1'.format(dataset, teacher_str)
            teacher.cpu()
            teacher.load_state_dict(torch.load(filename, map_location='cpu')['199'])
            teacher = teacher.to(device)

        else:
            teacher = None
        
        if 'wrn' in student_str:
            student_depth = int(student_str.split('-')[1])
            student_widen_factor = int(student_str.split('-')[2])
            student = cifar.WideResNet(depth=student_depth, widen_factor=student_widen_factor, num_classes=num_classes)
        elif 'res' in student_str:
            student_depth = int(student_str.split('-')[1])
            student_widen_factor = int(student_str.split('-')[2])
            student = cifar.ResNet(depth=student_depth, widen_factor=student_widen_factor, num_classes=num_classes)
    
    elif dataset == 'imagenet':
        bn_aff = False if 'nobn' in teacher_str else True
        shortcut = False if 'demo' in teacher_str else True

        if 'res' in teacher_str:
            teacher_depth = int(teacher_str.split('-')[1])
            if teacher_depth == 152:
                teacher = imagenet.resnet152(pretrained=True)
            elif teacher_depth == 50:
                teacher = imagenet.resnet50(pretrained=True)
            elif teacher_depth == 34:
                teacher = imagenet.resnet34(pretrained=True)
        else:
            teacher = None
        
        if 'res' in student_str:
            student_depth = int(student_str.split('-')[1])
            if student_depth == 152:
                student = imagenet.resnet152()
            elif student_depth == 50:
                student = imagenet.resnet50()
                
    elif dataset == 'tiny-imagenet':
        # teacher part will be modified
        if teacher_str is not None:
            if 'res' in teacher_str:
                teacher_depth = int(teacher_str.split('-')[1])
                if teacher_depth == 152:
                    teacher = imagenet.resnet152(num_classes=num_classes)
                elif teacher_depth == 50:
                    teacher = imagenet.resnet50(num_classes=num_classes)
                elif teacher_depth == 34:
                    teacher = imagenet.resnet34(num_classes=num_classes)
                    
            filename = './model_checkpoints/{}/None/{}/alp_0.1_T_1.0/random_highest_1.0_random_highest_1.0_seed1.t1'.format(dataset, teacher_str)
            teacher.cpu()
            teacher.load_state_dict(torch.load(filename, map_location='cpu')['199'])
            teacher = teacher.to(device)
        else:
            teacher = None
        
        if 'res' in student_str:
            student_depth = int(student_str.split('-')[1])
            if student_depth == 152:
                student = imagenet.resnet152(num_classes=num_classes)
            elif student_depth == 50:
                student = imagenet.resnet50(num_classes=num_classes)
            elif student_depth == 34:
                student = imagenet.resnet34(num_classes=num_classes)
        
    return teacher, student

def load_dataloader(dataset,
                    teacher,
                    mode,
                    batch_size,
                    root,
                    model_name,
                    cls_acq,
                    cls_order,
                    zeta,
                    sample_acq,
                    sample_order,
                    delta):
    
    if dataset == 'cifar10':
        dataloaders, _ = cifar_10_setter(teacher=teacher,
                                         mode=mode,
                                         batch_size=batch_size,
                                         root=root,
                                         model_name=model_name,
                                         cls_acq=cls_acq,
                                         cls_order=cls_order,
                                         zeta=zeta,
                                         sample_acq=sample_acq,
                                         sample_order=sample_order,
                                         delta=delta)
        
    elif dataset == 'cifar100':
        dataloaders, _ = cifar_100_setter(teacher=teacher,
                                          mode=mode,
                                          batch_size=batch_size,
                                          root=root,
                                          model_name=model_name,
                                          cls_acq=cls_acq,
                                          cls_order=cls_order,
                                          zeta=zeta,
                                          sample_acq=sample_acq,
                                          sample_order=sample_order,
                                          delta=delta)
    elif dataset == 'imagenet':
        dataloaders, _ = imagenet_setter(teacher=teacher,
                                         mode=mode,
                                         batch_size=batch_size,
                                         root=root,
                                         model_name=model_name,
                                         cls_acq=cls_acq,
                                         cls_order=cls_order,
                                         zeta=zeta,
                                         sample_acq=sample_acq,
                                         sample_order=sample_order,
                                         delta=delta)
    elif dataset == 'tiny-imagenet':
        dataloaders, _ = tiny_imagenet_setter(teacher=teacher,
                                         mode=mode,
                                         batch_size=batch_size,
                                         root=root,
                                         model_name=model_name,
                                         cls_acq=cls_acq,
                                         cls_order=cls_order,
                                         zeta=zeta,
                                         sample_acq=sample_acq,
                                         sample_order=sample_order,
                                         delta=delta)
        
    return dataloaders