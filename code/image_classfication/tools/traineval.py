import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from time import time
from tqdm import tqdm
from models import cifar, imagenet
from .losses import *

def make_log(model_name, dataloaders, num_epochs):
    log_filename = './results/' + model_name + '_log.csv'
    log_columns = ['train_loss', 'train_accuracy']
    if len(dataloaders['valid']) != 0:
        log_columns += ['valid_loss', 'valid_accuracy']
    log_columns += ['test_loss', 'test_accuracy']
    log_pd = pd.DataFrame(np.zeros([num_epochs, len(log_columns)]), columns=log_columns)
    return log_filename, log_pd

def train_model(model_name, device, dataloaders, teacher, student, optimizer, num_epochs, scheduler, criterion, args):
    if teacher is not None:
        if args.ensemble:
            for key in teacher.keys():
                teacher[key].to(device)
                teacher[key].eval()
                for p in teacher[key].parameters():
                    p.requires_grad = False
        else:
            teacher.to(device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
            
    student.to(device)
    
    if args.dataset != 'imagenet':
        log_filename, log_pd = make_log(model_name=model_name, dataloaders=dataloaders, num_epochs=num_epochs)
    state = {}
    current_state = copy.deepcopy(student.state_dict())
    for k, v in current_state.items():
        current_state[k] = v.cpu()
    state['init'] = copy.deepcopy(current_state)

    for epoch in tqdm(range(num_epochs)):
        student.train()
        epoch_log = []
        for i, data in enumerate(dataloaders['train']):
            image = data[0].type(torch.FloatTensor).to(device)
            label = data[1].type(torch.LongTensor).to(device)

            pred_label = student(image)
            if teacher is not None:
                if args.ensemble:
                    teacher_label = None
                    for key in teacher.keys():
                        if teacher_label == None:
                            teacher_label = teacher[key](image)
                        else:
                            teacher_label += teacher[key](image)
                else:
                    teacher_label = teacher(image)

            optimizer.zero_grad()
            
            if teacher is not None:
                if args.logit == 'l2_logit':
                    loss = mse_logit(pred_label, teacher_label)
                elif args.logit == 'smooth_logit':
                    loss = smooth_logit(pred_label, teacher_label)
                elif args.temperature > 150:
                    inf_gradient = inf_grad(pred_label, teacher_label, pred_label.shape[1], device)
                    pred_label.backward(args.alpha * inf_gradient, retain_graph=True)
                    loss = nn.CrossEntropyLoss()(pred_label, label) * (1 - args.alpha)
                else:
                    loss = criterion(pred_label, label, teacher_label, args.alpha, args.temperature)
            else:
                loss = criterion(pred_label, label)

            loss.backward()
            optimizer.step()
            
        if scheduler is not None:
            scheduler.step()
        
        if args.dataset != 'imagenet':
            loss_log, acc_log = eval_model(teacher, student, dataloaders['train'], criterion, args.alpha, args.temperature, device, args.ensemble)
            epoch_log += [loss_log, acc_log]

            if len(dataloaders['valid']) != 0:
                loss_log, acc_log = eval_model(teacher, student, dataloaders['valid'], criterion, args.alpha, args.temperature, device, args.ensemble)
                epoch_log += [loss_log, acc_log]

            loss_log, acc_log = eval_model(teacher, student, dataloaders['test'], criterion, args.alpha, args.temperature, device, args.ensemble)
            epoch_log += [loss_log, acc_log]
            log_pd.loc[epoch] = epoch_log
            log_pd.to_csv(log_filename)
        
        if epoch == num_epochs-1:
            current_state = copy.deepcopy(student.state_dict())
            for k, v in current_state.items():
                current_state[k] = v.cpu()
            state[str(epoch)] = copy.deepcopy(current_state)
        
        if args.save_all :
            current_state = copy.deepcopy(student.state_dict())
            for k, v in current_state.items():
                current_state[k] = v.cpu()
            state[str(epoch)] = copy.deepcopy(current_state)
            torch.save(state, './model_checkpoints/'+model_name+'.t1')

        
    torch.save(state, './model_checkpoints/'+model_name+'.t1')
    print ('./model_checkpoints/'+model_name+'.t1')
    
def eval_model(teacher, student, loader, criterion, alpha, temperature, device, ensemble):
    if teacher is not None:
        if ensemble:
            for key in teacher.keys():
                teacher[key].eval()
        else:
            teacher.eval()
    student.eval()
    losses = []
    for i, data in enumerate(loader):
        image = data[0].type(torch.FloatTensor).to(device)
        label = data[1].type(torch.LongTensor).to(device)
        
        pred_label = student(image)

        if teacher is not None:
            if ensemble:
                teacher_label = None
                for key in teacher.keys():
                    if teacher_label == None:
                        teacher_label = teacher[key](image)
                    else:
                        teacher_label += teacher[key](image)
            else:
                teacher_label = teacher(image)
            loss = criterion(pred_label, label, teacher_label, alpha, temperature)
        else:
            loss = criterion(pred_label, label)
        losses.append(loss.item())
        
        if i == 0:
            labels = label.cpu().detach().numpy()
            pred_labels = pred_label.cpu().detach().numpy()
        else:
            labels = np.concatenate((labels, label.cpu().detach().numpy()), axis=0, out=None)
            pred_labels = np.concatenate((pred_labels, pred_label.cpu().detach().numpy()), axis=0, out=None)
            
        image = image.cpu()
        label = label.cpu()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    pred_labels = np.argmax(pred_labels, axis=1)
    return np.mean(losses), np.sum(pred_labels==labels)/float(labels.size)