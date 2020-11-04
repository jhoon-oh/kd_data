import os
import random
from collections import defaultdict
import torch
import scipy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tools.data_setter import cifar_100_setter
from tools.generator import generate_sample_info
from models import cifar, imagenet

def get_entropy(img, model):
    probs = torch.softmax(model(img), dim=1)
    log_probs = torch.log(probs)
    entropy = -torch.sum(probs*log_probs, dim = 1 )
    pred_label = torch.argmax(probs).item()
    return pred_label, entropy

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def plot_accuracy(student_path):
    result_path = "results/" +  "/".join(student_path.split("/")[2:])
    result_files = os.listdir(result_path)
    result_files = [i for i in result_files if i[-3:] == 'csv']

    for idx , f in enumerate(result_files):
        if idx == 0 :
            df = pd.read_csv(result_path +"/"+result_files[idx])
            train_series = df["train_accuracy"]
            test_series = df["test_accuracy"]
        else :
            df = pd.read_csv(result_path +"/"+result_files[idx])
            train_series += df["train_accuracy"]
            test_series += df["test_accuracy"]
    train_series /= idx+1
    test_series /= idx+1

    fig = go.Figure()
    epochs = list(range(200))
    fig.add_trace(go.Scatter(x=epochs, y=train_series, name="Train accuracy"))
    fig.add_trace(go.Scatter(x=epochs, y=test_series, name= "Test accuracy"))
    fig.update_layout(
        showlegend=True,
        annotations=[
            dict(
                x=199,
                y=train_series.iloc[-1],
                xref="x",
                yref="y",
                text=str(round(train_series.iloc[-1],3)),
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=-40
            ),
            dict(
                x=199,
                y=test_series.iloc[-1],
                xref="x",
                yref="y",
                text=str(round(test_series.iloc[-1],3)),
                showarrow=True,
                arrowhead=7,
                ax=0,
                ay=40
            )
        ]
    )
    fig.show()


def histogram_tld(teacher_df, student_df):
    fig =make_subplots(rows=1, cols=2, subplot_titles=("Train ", "Test"))

    df_plot = teacher_df[teacher_df.data_type=='train'].copy()
    df_plot.sort_values("tld", ascending=False ,inplace=True)
    df_plot.reset_index(inplace=True)

    fig.add_trace(go.Histogram(x=df_plot.tld, name="Teacher", opacity=0.8, histnorm='percent'), row=1, col=1)

    df_plot = student_df[student_df.data_type=='train'].copy()
    df_plot.sort_values("tld", ascending=False ,inplace=True)
    df_plot.reset_index(inplace=True)
    fig.add_trace(go.Histogram(x=df_plot.tld, name="Student", opacity=0.8, histnorm='percent'), row=1, col=1)
    

    df_plot = teacher_df[teacher_df.data_type=='test'].copy()
    df_plot.sort_values("tld", ascending=False ,inplace=True)
    df_plot.reset_index(inplace=True)
    fig.add_trace(go.Histogram(x=df_plot.tld, name="Teacher", opacity=0.8, histnorm='percent'), row=1, col=2)

    df_plot = student_df[student_df.data_type=='test'].copy()
    df_plot.sort_values("tld", ascending=False ,inplace=True)
    df_plot.reset_index(inplace=True)
    fig.add_trace(go.Histogram(x=df_plot.tld, name="Student", opacity=0.8, histnorm='percent'), row=1, col=2)

    fig.update_layout(title="Teacher")
    
    fig.show()

def quantile_tld(teacher_df, student_df, num_threshold=4, sep=False,data_type="train"):

    teacher_df_plot = teacher_df.copy()
    student_df_plot = student_df.copy()
    teacher_df_plot, student_df_plot = make_quantile_column(teacher_df_plot,\
                                                            student_df_plot, num_threshold=num_threshold)
    student_df_plot.sort_values("teacher_class", inplace=True)
    if sep:
        fig = go.Figure()
        grps = student_df_plot.groupby("teacher_class")
        for idx, g in enumerate(grps.groups.keys()):
            gr = grps.get_group(g)
            fig.add_trace(go.Histogram(x=gr[gr.data_type==data_type].tld, \
                                       opacity=0.5, name=idx, histnorm='percent'))
        fig.update_xaxes(range=[-15,25])
        fig.update_layout(barmode="overlay",title="Student {} TLD wrt teacher quantile".format(data_type))
        
        fig.show()
    else :
        fig = px.histogram(student_df_plot[student_df_plot.data_type==data_type], x="tld", color="teacher_class",  histnorm='percent' , title="Student {} TLD wrt teacher quantile".format(data_type))
        fig.update_xaxes(range=[-15, 25])
        fig.show()
    
def quantile_acc(teacher_df, student_df, num_threshold=4 ,sub=False):
    teacher_df_plot = teacher_df.copy()
    student_df_plot = student_df.copy()
    teacher_df_plot, student_df_plot = make_quantile_column(teacher_df_plot,\
                                                            student_df_plot, num_threshold ,sub)
    fig =make_subplots(rows=1, cols=2, subplot_titles=("Student", "Teacher"))
    t_list =[0,1,2,3]
    accs = defaultdict(list)
    for dataset_type in ["train", "test"]:
        for t in t_list:
            acc_df = student_df_plot[(student_df_plot.teacher_class == t) & (student_df_plot.data_type== dataset_type)]
            accs[dataset_type].append(acc_df.accuracy.sum()/len(acc_df))


    fig.add_trace(go.Scatter(x=t_list, y=accs['train'], name="Train"))
    fig.add_trace(go.Scatter(x=t_list, y=accs["test"], name="Test"))

    accs = defaultdict(list)
    for dataset_type in ["train", "test"]:
        for t in t_list:
            acc_df = teacher_df_plot[(teacher_df_plot.teacher_class == t) & (teacher_df_plot.data_type== dataset_type)]
            accs[dataset_type].append(acc_df.accuracy.sum()/len(acc_df))


    fig.add_trace(go.Scatter(x=t_list, y=accs['train'], name="Train"), row=1,col=2)
    fig.add_trace(go.Scatter(x=t_list, y=accs["test"], name="Test"), row=1,col=2)

    fig.update_yaxes(range=(0.2,1))
    fig.update_layout(title="ACC wrt Threshold")
    fig.show()    
    
def make_quantile_column(teacher_df,student_df, num_threshold=4,sub=False):
    teacher_df_plot = teacher_df.copy()
    student_df_plot = student_df.copy()

    teacher_df_plot["sub"] = teacher_df_plot.tld - student_df_plot.tld

    qs =np.linspace(0,1, num_threshold+1)

    teacher_class = pd.Series(dtype=int)
    for data_type in ["train", "test"]:
        thresholds = []
        teacher_df_plot_by_datatype = teacher_df_plot[teacher_df_plot.data_type==data_type]
        student_df_plot_by_datatype = student_df_plot[student_df_plot.data_type==data_type]


        if sub:
            for q in qs:
                thresholds.append(teacher_df_plot_by_datatype["sub"].quantile(q))
        else :
            for q in qs:
                thresholds.append(teacher_df_plot_by_datatype.tld.quantile(q))
        
        thresholds[0] = -9999
        def make_quantile_class(x):
            tmp = x > np.array(thresholds)
            clas = len(np.where(tmp)[0])
            return clas-1

        if sub:
            teacher_class = teacher_class.append(teacher_df_plot_by_datatype['sub'].apply(lambda x : make_quantile_class(x)))

        else :
            teacher_class = teacher_class.append(teacher_df_plot_by_datatype.tld.apply(lambda x : make_quantile_class(x)))
        
    student_df_plot["teacher_class"] = teacher_class
    teacher_df_plot["teacher_class"] = teacher_class
    return teacher_df_plot, student_df_plot

def get_dataframe(teacher, student, teacher_checkpoint, student_checkpoint, nets=False, epoch="199",seed=0 ,device="cuda:1"):
    device = torch.device(device)
    teacher_epoch = epoch

    student.cpu()
    student.load_state_dict(torch.load(student_checkpoint, map_location='cpu')[epoch])
    student = student.to(device)
    student.eval()

    if nets :
        teacher.cpu()
        teacher.load_state_dict(torch.load(teacher_checkpoint, map_location='cpu')['nets']['199'])
        teacher = teacher.to(device)
        teacher.eval()
    else :
        teacher.cpu()
        teacher.load_state_dict(torch.load(teacher_checkpoint, map_location='cpu')['199'])
        teacher = teacher.to(device)
        teacher.eval()
        
    generate_sample_info(teacher, dataset="cifar100", root='./data/',
                    model_name="/".join(student_checkpoint.split('/')[2:])
                    ,device=device)
    
    set_seed(seed)
    dataloaders, dataset_size = cifar_100_setter(teacher=teacher,
                                             mode="crop",
                                             batch_size=128,
                                             root='./data/',
                                             model_name="/".join(student_checkpoint.split('/')[2:]),
                                             cls_acq='random',
                                             per_class=True,
                                             cls_lower_qnt=0.0,
                                             cls_upper_qnt=1.0,
                                             sample_acq='random',
                                             sample_lower_qnt=0.0,
                                             sample_upper_qnt=1.0)
    
    teacher = teacher.to(device)
    student = student.to(device)

    
    set_seed(seed)
    for i, data in enumerate(dataloaders['train']):
        image = data[0].type(torch.FloatTensor).to(device)
        label = data[1].type(torch.LongTensor).to(device) 
        if i==0:
            labels = label.cpu()
            teachers_labels = teacher(image).detach().cpu()
            entropys = get_entropy(image, teacher)[1].detach().cpu()
        else:
            labels = torch.cat([labels, label.cpu()])
            teachers_labels = torch.cat([teachers_labels, teacher(image).detach().cpu()], dim=0)
            entropy = get_entropy(image, teacher)[1].detach().cpu()
            entropys = torch.cat([entropys, entropy], dim=0)

    accuracy = labels == torch.argmax(teachers_labels, dim=1)    
    values, _ = torch.topk(teachers_labels, k=3, dim=1)
    top1_top2_list = values[:,0]-values[:,1]
    gt_list=[]
    for i in range(len(teachers_labels)):
        gt_list.append(teachers_labels[i,labels[i].item()].item())
    gt_list = torch.Tensor(gt_list)
    top1_gt_list = values[:,0] - gt_list
    teacher_df = pd.DataFrame({"label":labels.numpy(), "top1_top2":top1_top2_list.numpy(),
                             "top1_gt":top1_gt_list.numpy(), "accuracy":accuracy.numpy(),
                              "entropy":entropys.numpy(), "data_type": ["train"]*labels.numpy().shape[0],
                              "epoch":[teacher_epoch]*labels.numpy().shape[0],
                              "data_index": list(range(i+1))})
    for i, data in enumerate(dataloaders['test']):
        image = data[0].type(torch.FloatTensor).to(device)
        label = data[1].type(torch.LongTensor).to(device)
        if i==0:
            labels = label.cpu()
            teachers_labels = teacher(image).detach().cpu()
            entropys = get_entropy(image, teacher)[1].detach().cpu()
        else:
            labels = torch.cat([labels, label.cpu()])
            teachers_labels = torch.cat([teachers_labels, teacher(image).detach().cpu()], dim=0)
            entropy = get_entropy(image, teacher)[1].detach().cpu()
            entropys = torch.cat([entropys, entropy], dim=0)

    accuracy = labels == torch.argmax(teachers_labels, dim=1)    
    values, _ = torch.topk(teachers_labels, k=3, dim=1)
    top1_top2_list = values[:,0]-values[:,1]
    gt_list=[]
    for i in range(len(teachers_labels)):
        gt_list.append(teachers_labels[i,labels[i].item()].item())
    gt_list = torch.Tensor(gt_list)
    top1_gt_list = values[:,0] - gt_list

    teacher_df =  teacher_df.append(pd.DataFrame({"label":labels.numpy(), "top1_top2":top1_top2_list.numpy(),
                             "top1_gt":top1_gt_list.numpy(), "accuracy":accuracy.numpy(),
                              "entropy":entropys.numpy(), "data_type": ["test"]*labels.numpy().shape[0],
                              "epoch":[teacher_epoch]*labels.numpy().shape[0],
                              "data_index": list(range(i+1))}))

    tld_list = []
    for i,x in enumerate(teacher_df.to_dict("records")):
        if x["accuracy"] == 0 :
            tld_list.append(-x["top1_gt"])
        else :
            tld_list.append(x["top1_top2"])

    teacher_df["tld"] = tld_list
    teacher_df.head()
    
    
    set_seed(seed)
    for i, data in enumerate(dataloaders['train']):
        image = data[0].type(torch.FloatTensor).to(device)
        label = data[1].type(torch.LongTensor).to(device)

        if i==0:
            labels = label.cpu()
            students_labels = student(image).detach().cpu()
            entropys = get_entropy(image, student)[1].detach().cpu()
        else:
            labels = torch.cat([labels, label.cpu()])
            students_labels = torch.cat([students_labels, student(image).detach().cpu()], dim=0)
            entropy = get_entropy(image, student)[1].detach().cpu()
            entropys = torch.cat([entropys, entropy], dim=0)

    accuracy = labels == torch.argmax(students_labels, dim=1)    
    values, _ = torch.topk(students_labels, k=3, dim=1)
    top1_top2_list = values[:,0]-values[:,1]

    gt_list=[]
    for i in range(len(students_labels)):
        gt_list.append(students_labels[i,labels[i].item()].item())
    gt_list = torch.Tensor(gt_list)

    top1_gt_list = values[:,0] - gt_list

    student_df = pd.DataFrame({"label":labels.numpy(), "top1_top2":top1_top2_list.numpy(),
                             "top1_gt":top1_gt_list.numpy(), "accuracy":accuracy.numpy(),
                              "entropy":entropys.numpy(), "data_type": ["train"]*labels.numpy().shape[0],
                              "epoch":[epoch]*labels.numpy().shape[0],
                              "data_index": list(range(i+1))})


    for i, data in enumerate(dataloaders['test']):
        image = data[0].type(torch.FloatTensor).to(device)
        label = data[1].type(torch.LongTensor).to(device)

        if i==0:
            labels = label.cpu()
            students_labels = student(image).detach().cpu()
            entropys = get_entropy(image,student)[1].detach().cpu()
        else:
            labels = torch.cat([labels, label.cpu()])
            students_labels = torch.cat([students_labels, student(image).detach().cpu()], dim=0)
            entropy = get_entropy(image, student)[1].detach().cpu()
            entropys = torch.cat([entropys, entropy], dim=0)



    accuracy = labels == torch.argmax(students_labels, dim=1)    
    values, _ = torch.topk(students_labels, k=3, dim=1)
    top1_top2_list = values[:,0]-values[:,1]

    gt_list=[]
    for i in range(len(students_labels)):
        gt_list.append(students_labels[i,labels[i].item()].item())
    gt_list = torch.Tensor(gt_list)

    top1_gt_list = values[:,0] - gt_list

    student_df =  student_df.append(pd.DataFrame({"label":labels.numpy(), "top1_top2":top1_top2_list.numpy(),
                             "top1_gt":top1_gt_list.numpy(), "accuracy":accuracy.numpy(),
                              "entropy":entropys.numpy(), "data_type": ["test"]*labels.numpy().shape[0],
                              "epoch":[epoch]*labels.numpy().shape[0],
                              "data_index": list(range(i+1))}))

    tld_list = []
    for i,x in enumerate(student_df.to_dict("records")):
        if x["accuracy"] == 0 :
            tld_list.append(-x["top1_gt"])
        else :
            tld_list.append(x["top1_top2"])
    student_df["tld"] = tld_list
    student_df.head()
    
    teacher_df.reset_index(drop=True, inplace=True)
    student_df.reset_index(drop=True, inplace=True)
    
    return teacher_df, student_df

