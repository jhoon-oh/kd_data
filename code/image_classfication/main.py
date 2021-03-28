#!/usr/bin/env python3
import argparse
import torch
import os
import numpy as  np
import random

from tools.loader import load_model, load_dataloader
from tools.generator import generate_sample_info
from tools.losses import cr_loss, kd_loss
from tools.traineval import train_model

def main(args):
    # make experiment reproducible
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print(args.sample_upper_qnt)

    model_name = '{}'.format(args.dataset) + \
                 '/{}/{}'.format(str(args.teacher), str(args.student)) + \
                 '/alp_{}_T_{}'.format(str(args.alpha), str(args.temperature)) + \
                 '/{}_{}-{}_{}_{}-{}_seed{}_{}'.format(args.cls_acq,
                                                       str(args.cls_lower_qnt),
                                                       str(args.cls_upper_qnt),
                                                       args.sample_acq,
                                                       str(args.sample_lower_qnt),
                                                       str(args.sample_upper_qnt),
                                                       str(args.seed),
                                                       str(args.logit))
    
    if not args.per_class:
        model_name += '_noclas'
        
    result_path = './results'
    checkpoint_path = './model_checkpoints'
    model_path = model_name.split('/')
    
    for tmp_path in [result_path, checkpoint_path]:
        if not os.path.isdir(tmp_path):
            os.mkdir(tmp_path)
        next_path = os.path.join(tmp_path, model_path[0])
        if not os.path.isdir(next_path):
            os.mkdir(next_path)
        next_path = os.path.join(next_path, model_path[1])
        if not os.path.isdir(next_path):
            os.mkdir(next_path)
        next_path = os.path.join(next_path, model_path[2])
        if not os.path.isdir(next_path):
            os.mkdir(next_path)
        next_path = os.path.join(next_path, model_path[3])
        if not os.path.isdir(next_path):
            os.mkdir(next_path)
    
    device = torch.device(args.device)
    teacher, student = load_model(teacher_str=args.teacher,
                                  student_str=args.student,
                                  dataset=args.dataset,
                                  device=device, ensemble=args.ensemble)
    
    generate_sample_info(teacher=teacher[args.teacher.split(',')[0]] if args.ensemble else teacher,
                         dataset=args.dataset,
                         root=args.root,
                         model_name=model_name,
                         device=device)
    
    dataloaders = load_dataloader(dataset=args.dataset,
                                  teacher=teacher[args.teacher.split(',')[0]] if args.ensemble else teacher,
                                  mode=args.mode,
                                  batch_size=args.batch_size,
                                  root=args.root,
                                  model_name=model_name,
                                  per_class=args.per_class,
                                  cls_acq=args.cls_acq,
                                  cls_lower_qnt=args.cls_lower_qnt,
                                  cls_upper_qnt=args.cls_upper_qnt,
                                  sample_acq=args.sample_acq,
                                  sample_lower_qnt=args.sample_lower_qnt,
                                  sample_upper_qnt=args.sample_upper_qnt)
    
    batch_params = [module for module in student.parameters() if module.ndimension() == 1]
    other_params = [module for module in student.parameters() if module.ndimension() > 1]
    optimizer = torch.optim.SGD([{'params': batch_params, 'weight_decay': 0},
                                 {'params': other_params, 'weight_decay': args.weight_decay}],
                                 lr=args.lr,
                                 momentum=args.momentum,
                                 nesterov=args.nesterov)
    num_epochs = args.num_epochs
    if args.scheduler is None:
        scheduler = None
    elif args.scheduler == 'multistep':
        milestones = [int(epoch) for epoch in args.milestones.split(',')]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=milestones,
                                                         gamma=args.gamma)
    criterion = kd_loss if teacher is not None else cr_loss
    
    train_model(model_name=model_name,
                device=device,
                dataloaders=dataloaders,
                teacher=teacher,
                student=student,
                optimizer=optimizer,
                num_epochs=num_epochs,
                scheduler=scheduler,
                criterion=criterion,
                args=args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge Distillation')
    parser.add_argument('--device', default='cuda:0', type=str, help='which device to use')
    
    parser.add_argument('--teacher', default=None, type=str, help='teacher model (string)')
    parser.add_argument('--student', default=None, type=str, help='student model (string)')
    parser.add_argument('--dataset', default='cifar100', type=str, help='which dataset to use')
    parser.add_argument('--ensemble', help='whether to use ensemble', action='store_true')
    
    parser.add_argument('--mode', default='crop', type=str, help='augmentation strategy (vanilla, flip, crop, fastauto, auto)')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--root', default='./data/', type=str, help='directory for dataset')

    # dataset setting-related parser
    parser.add_argument('--per_class', help='the ratio per class is considered or not', action='store_true')
    parser.add_argument('--cls_acq', default='random', type=str, help='class selection function: random or entropy')
    parser.add_argument('--cls_lower_qnt', default=0.0, type=float, help='class selection lower quantile')
    parser.add_argument('--cls_upper_qnt', default=1.0, type=float, help='class selection upper quantile')
    parser.add_argument('--sample_acq', default='random', type=str, help='sample selection function: random or entropy')
    parser.add_argument('--sample_lower_qnt', default=0.0, type=float, help='sample selection lower quantile')
    parser.add_argument('--sample_upper_qnt', default=1.0, type=float, help='sample selection upper quantile')
    
    parser.add_argument('--seed', default=1, type=int, help='seed')
    
    # model training-related parser
    parser.add_argument('--optimizer', default='sgd', type=str, help='which optimizer to use')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='nesterov')
    parser.add_argument('--nesterov', help='whether to use nesterov', action='store_true')
    
    parser.add_argument('--num_epochs', default=200, type=int, help='epochs #')
    parser.add_argument('--scheduler', default='multistep', type=str, help='which scheduler to use')
    parser.add_argument('--milestones', default='100,150', type=str, help='milestone epochs with commas')
    parser.add_argument('--gamma', default=0.1, type=float, help='scheduler ratio')
    
    parser.add_argument('--alpha', default=0.1, type=float, help='alpha of kd loss')
    parser.add_argument('--temperature', default=1.0, type=float, help='temperature of kd loss')
    parser.add_argument('--save_all', help='save weights?', action='store_true')
    parser.add_argument('--logit', default='none', type=str, choices=['none', 'l2_logit', 'smooth_logit'],  help='logit_loss')
        
    args = parser.parse_args()
    main(args)