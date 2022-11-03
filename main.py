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

data  = Rfile_train(PATH="unlabeled")
test ,y_test= Rfile_test(PATH="test")

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
test_data = img_Dataset(test,label=y_test,transform=transform)


train_loader1 = DataLoader(train_data1 , batch_size=64  ,  shuffle = False)
train_loader2 = DataLoader(train_data2 , batch_size=64 ,  shuffle = False)
test_loader =  DataLoader(test_data , batch_size=64 ,  shuffle = False)

print(len(train_loader1),len(test_loader))

gpu0_bsz = 64 / 2
acc_grad = 1
model = restnet50()
model = BalancedDataParallel(gpu0_bsz // acc_grad, model, dim=0).cuda()

optimzier = Optimizer(name = "Lookahead",model = model  ,lr = 0.01, k=5 , alpha = 0.5)

criterion = ntX

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimzier, mode='min', factor=0.5, patience=2, min_lr=0.000001)

best_model_path =  train(model,train_loader1,train_loader2,optimzier ,criterion,3,scheduler,600)


criterion = nn.CrossEntropyLoss()

optimizer_ft = Optimizer(name = "Lookahead" , model = model ,lr = 0.01 ,k=5 , alpha = 0.5)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.5, patience=2, min_lr=0.000001)
best_model_path_ft = fine_tune(model, best_model_path ,test_loader,optimizer_ft,criterion,scheduler,4,200)

# import gc
# gc.collect()
# torch.cuda.empty_cache()

run_pred(model , best_model_path_ft , test_loader)
