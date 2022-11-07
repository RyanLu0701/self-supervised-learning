import torchvision.transforms as transforms
from IPython.display import display
from PIL import Image
import os
import numpy as np
import pandas as pd
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import datasets, models
from torch.utils.data import Dataset, DataLoader
from Tool.nt_xent import nt_xent as ntX
from Tool.readfile import readfile_train as Rfile_train
from Tool.readfile import readfile_test as Rfile_test
from Tool.Lookahead import Lookahead
from Tool.data_parallel_my_v2 import BalancedDataParallel
from Tool.KNN import KNN
from model import restnet50
from Run_test import img_Dataset,train,fine_tune,Optimizer,run_pred
import argparse

###
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",default = 64, help="data batch_size",type = int )
parser.add_argument("--lr",default = 0.01 ,help="learning rate ",type = float )
parser.add_argument("--optimizer",default = "Lookahead" , help="Lookahead  or  Adam " )
parser.add_argument("--k" ,            default = 5 , help = "parameter of Lookahead , if use adam make it None ",type = int )
parser.add_argument("--alpha" ,        default = 0.5 , help = "parameter of Lookahead , if use adam make it None ",type = float)
parser.add_argument("--scheduler_factor",default = 0.5 , help = "scheduler factor ",type = float)
parser.add_argument("--scheduler_patience",default = 3 , help = "scheduler patience",type = int)
parser.add_argument("--scheduler_min_lr",default = 0.0000001, help = "scheduler min lr",type = float)
parser.add_argument("--patience",default = 3 , help = "patience for train",type = int)
parser.add_argument("--patience_ft",default = 3 , help = "patience for fine tune",type = int)
parser.add_argument("--epochs",default = 200 , help = "epochs for train",type = int)
parser.add_argument("--epochs_ft",default = 100 , help = "epochs for fine tune",type = int)
parser.add_argument("--lr_ft",default = 0.01 , help = "epochs for fine tune",type = float)
parser.add_argument("--k_ft",default = 5 , help = "epochs for fine tune",type = int)
parser.add_argument("--alpha_ft",default = 0.5 , help = "epochs for fine tune",type = float)

args = parser.parse_args()

###
data          = Rfile_train(PATH="unlabeled")
test , y_test = Rfile_test(PATH="test")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Pad(padding=(0, 0, 8, 16), fill=128),
    transforms.RandomHorizontalFlip(p=0.9),
    transforms.RandomEqualize(p=0.5),
    transforms.ToTensor(),

])


train_data1 = img_Dataset(data,label=None,transform=transform)
train_data2 = img_Dataset(data,label=None,transform=transform)
test_data   = img_Dataset(test,label=y_test,transform=transform)


train_loader1 =   DataLoader(train_data1 , batch_size=args.batch_size  ,  shuffle = False)
train_loader2 =   DataLoader(train_data2 , batch_size=args.batch_size  ,  shuffle = False)
test_loader   =   DataLoader(test_data   , batch_size=args.batch_size  ,  shuffle = False)

#device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

model = restnet50()
model = model.to(device)

if args.optimizer =="Lookahead":
    optimizer = Optimizer(name = "Lookahead",model = model  ,lr = args.lr, k=args.k , alpha = args.alpha)
else :
    optimizer = Optimizer(name = "Lookahead",model = model  ,lr = args.lr)

criterion = ntX

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)

best_model_path =  train(model,train_loader1,train_loader2,optimizer ,criterion,args.patience,scheduler,args.epochs,device)

criterion = nn.CrossEntropyLoss()

if args.optimizer =="Lookahead":

    optimizer_ft = Optimizer(name = "Lookahead",model = model  ,lr = args.lr_ft, k=args.k_ft , alpha = args.alpha_ft)

else :

    optimizer_ft = Optimizer(name = "Lookahead",model = model  ,lr = args.lr_ft)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=args.scheduler_factor, patience=args.scheduler_patience, min_lr=args.scheduler_min_lr)
best_model_path_ft = fine_tune(model, best_model_path ,test_loader,y_test,optimizer_ft,criterion,scheduler,args.patience_ft,args.epochs_ft,device)

# import gc
# gc.collect()
# torch.cuda.empty_cache()

run_pred(model , best_model_path_ft ,y_test, test_loader,device)
