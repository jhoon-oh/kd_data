import torch
from models import cifar, imagenet

from .data_setter import *

def load_model(teacher_str, student_str, dataset, device, ensemble):
    # teacher_str, student_str: string ex) wrn-28-4, res-28-4
    # nobn: the affine parameters in bn get False
    # demo: no shortcut
    
    if dataset == 'cifar10' or dataset == 'svhn':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'tiny-imagenet':
        num_classes = 200
    elif dataset == 'imagenet':
        num_classes = 1000
    
    if dataset == 'cifar10' or dataset == 'cifar100' or dataset == 'svhn':
        if teacher_str is not None:
            bn_aff = False if 'nobn' in teacher_str else True
            shortcut = False if 'demo' in teacher_str else True
            
            if ensemble:
                teacher_cand = teacher_str.split(',')
                teacher = {}
                for teacher_str_cand in teacher_cand:
                    if 'wrn' in teacher_str_cand:
                        teacher_depth = int(teacher_str_cand.split('-')[1])
                        teacher_widen_factor = int(teacher_str_cand.split('-')[2])
                        teacher_tmp = cifar.WideResNet(depth=teacher_depth, widen_factor=teacher_widen_factor, num_classes=num_classes, bn_aff = bn_aff, shortcut = shortcut)
                    filename = './model_checkpoints/{}/None/{}/alp_0.1_T_1.0/random_0.0-1.0_random_0.0-1.0_seed9999_none_noclas.t1'.format(dataset, teacher_str_cand)
                    teacher_tmp.cpu()
                    teacher_tmp.load_state_dict(torch.load(filename, map_location='cpu')['199'])
                    teacher_tmp = teacher_tmp.to(device)
                    teacher[teacher_str_cand] = teacher_tmp
            else:
                if 'wrn' in teacher_str:
                    teacher_depth = int(teacher_str.split('-')[1])
                    teacher_widen_factor = int(teacher_str.split('-')[2])
                    teacher = cifar.WideResNet(depth=teacher_depth, widen_factor=teacher_widen_factor, num_classes=num_classes, bn_aff = bn_aff, shortcut = shortcut)
                elif 'res' in teacher_str:
                    teacher_depth = int(teacher_str.split('-')[1])
                    teacher_widen_factor = int(teacher_str.split('-')[2])
                    teacher = cifar.ResNet(depth=teacher_depth, width=teacher_widen_factor, num_classes=num_classes, bn_aff = bn_aff, shortcut = shortcut)         

                filename = './model_checkpoints/{}/None/{}/alp_0.1_T_1.0/random_0.0-1.0_random_0.0-1.0_seed9999_none_noclas.t1'.format(dataset, teacher_str)
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
                    
            filename = './model_checkpoints/{}/None/{}/alp_0.1_T_1.0/random_highest_1.0_random_highest_1.0_seed1_none.t1'.format(dataset, teacher_str)
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
                    per_class,
                    cls_acq,
                    cls_lower_qnt,
                    cls_upper_qnt,
                    sample_acq,
                    sample_lower_qnt,
                    sample_upper_qnt):
    
    if dataset == 'cifar10':
        dataloaders, _ = cifar_10_setter(teacher=teacher,
                                         mode=mode,
                                         batch_size=batch_size,
                                         root=root,
                                         model_name=model_name,
                                         per_class=per_class,
                                         cls_acq=cls_acq,
                                         cls_lower_qnt=cls_lower_qnt,
                                         cls_upper_qnt=cls_upper_qnt,
                                         sample_acq=sample_acq,
                                         sample_lower_qnt=sample_lower_qnt,
                                         sample_upper_qnt=sample_upper_qnt)
        
    elif dataset == 'svhn':
        dataloaders, _ = svhn_setter(teacher=teacher,
                                         mode=mode,
                                         batch_size=batch_size,
                                         root=root,
                                         model_name=model_name,
                                         per_class=per_class,
                                         cls_acq=cls_acq,
                                         cls_lower_qnt=cls_lower_qnt,
                                         cls_upper_qnt=cls_upper_qnt,
                                         sample_acq=sample_acq,
                                         sample_lower_qnt=sample_lower_qnt,
                                         sample_upper_qnt=sample_upper_qnt)
        
    elif dataset == 'cifar100':
        dataloaders, _ = cifar_100_setter(teacher=teacher,
                                          mode=mode,
                                          batch_size=batch_size,
                                          root=root,
                                          model_name=model_name,
                                          per_class=per_class,
                                          cls_acq=cls_acq,
                                          cls_lower_qnt=cls_lower_qnt,
                                          cls_upper_qnt=cls_upper_qnt,
                                          sample_acq=sample_acq,
                                          sample_lower_qnt=sample_lower_qnt,
                                          sample_upper_qnt=sample_upper_qnt)
    elif dataset == 'imagenet':
        dataloaders, _ = imagenet_setter(teacher=teacher,
                                         mode=mode,
                                         batch_size=batch_size,
                                         root=root,
                                         model_name=model_name,
                                         per_class=per_class,
                                         cls_acq=cls_acq,
                                         cls_lower_qnt=cls_lower_qnt,
                                         cls_upper_qnt=cls_upper_qnt,
                                         sample_acq=sample_acq,
                                         sample_lower_qnt=sample_lower_qnt,
                                         sample_upper_qnt=sample_upper_qnt)
    elif dataset == 'tiny-imagenet':
        dataloaders, _ = tiny_imagenet_setter(teacher=teacher,
                                         mode=mode,
                                         batch_size=batch_size,
                                         root=root,
                                         model_name=model_name,
                                         per_class=per_class,
                                         cls_acq=cls_acq,
                                         cls_lower_qnt=cls_lower_qnt,
                                         cls_upper_qnt=cls_upper_qnt,
                                         sample_acq=sample_acq,
                                         sample_lower_qnt=sample_lower_qnt,
                                         sample_upper_qnt=sample_upper_qnt)
        
    return dataloaders