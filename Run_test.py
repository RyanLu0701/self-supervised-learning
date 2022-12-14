import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Tool.Lookahead  import Lookahead
import time
from Tool.KNN import KNN
import numpy as np
class img_Dataset(Dataset):
    def __init__(self,imgs,label,transform=None, target_transform=None):

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.label = label

    def __getitem__(self, index):

        fn = self.imgs[index]


        if self.transform is not None:
            fn = self.transform(fn)

        if self.label is not None:
            label = self.label[index]
            return fn,label
        else:
            return fn

    def __len__(self):
        return len(self.imgs)


def Optimizer(name,lr,model,k=None,alpha=None):

    if name=="Adam":

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        return optimizer

    elif name =="Lookahead":

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        if k == None or alpha == None:

            raise ValueError("k or alpha is None")

        else:

            lookahead = Lookahead(optimizer, k=k, alpha=alpha)

        return lookahead

    else:

        raise ValueError("No such Optimizer name, use adam or Lookahead")



def train(net, trainLoader1,trainLoader2, optimizer, criterion,patience,scheduler, epochs,device ):

    time_start = time.time()
    time_avg   = []

    testAccuracy  = 0
    trigger_times = 0
    last_loss     = 100000

    best_loss     = 100000
    for i in range(0,epochs+1):

        net.train()

        totalLoss    = 0



        for idx ,data in  enumerate(zip(trainLoader1,trainLoader2)):

            data_1 = data[0].to(device)
            data_2 = data[1].to(device)

            optimizer.zero_grad()


            data_1 = data_1.float()
            data_2 = data_2.float()


            output_1    =   net(data_1)
            output_2    =   net(data_2)


            loss        =   criterion(output_1 , output_2)
            totalLoss   +=  loss.item()

            scheduler.step(loss.item())
            loss.backward()
            optimizer.step()

        remain_time = (((time.time()-time_start)/60)/(i+1))*(epochs-(i+1))

        time_avg.append(remain_time)

        mean_time = round(sum(time_avg[-5:])/5,3)
        print(f"Epoch [{epochs}/{i+1}] , total loss = {totalLoss/(len(trainLoader1.dataset)/trainLoader1.batch_size)} , estimate finish time : {mean_time}  min")



        current_loss = totalLoss/(len(trainLoader1.dataset)/trainLoader1.batch_size)




        if current_loss > last_loss:

            trigger_times += 1

            print('trigger times:', trigger_times)

            if trigger_times >= patience:

                print(f"Early stopping! at {i+1} :)) ")

                return bestModel

        else:

            if trigger_times>0:

                print("reset trigger time to 0")

            trigger_times =0


        last_loss = current_loss



        if best_loss >= current_loss:

            best_loss = current_loss
            bestModel = net

            print("Saving best model")

            torch.save(bestModel.state_dict(), "model_train/epoch"+str(i+1)+"_"+str(best_loss)+".pt")



    print("Complete training :)")


    return bestModel


def valid_test(model , test_loader,y_test,device):

    model.eval()

    num  = torch.zeros(500 ,512)

    count =0

    with torch.no_grad():

        for test_samples,_ in test_loader:

            test_samples = test_samples.to(device)

            test_outputs = model(test_samples)

            for i in range(len(test_outputs)):

                num[count] = test_outputs[i]

                count +=1

    acc = KNN(num , torch.tensor(y_test), batch_size=32)

    return acc

def fine_tune(net ,test_loader,y_test,optimizer,criterion,scheduler,patience,epochs,device):



    trigger_times = 0
    last_acc      = 0
    best_acc      = 0

    time_avg = []
    time_start = time.time()

    for i in range(0,epochs+1):

        net.train()

        totalLoss    = 0
        accuracy     = 0

        for idx ,(data,label) in  enumerate(test_loader):

            data  = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()

            data_= data.float()

            output    =   net(data, label = True)

            loss          = criterion(output , label)

            Total_output  = torch.argmax(output, dim=1)

            accuracy   +=  (Total_output == label).sum().item()

            totalLoss  +=  loss.item()

            scheduler.step(loss.item())
            loss.backward()
            optimizer.step()


        valid_acc = valid_test(model = net , test_loader = test_loader , y_test = y_test , device = device)

        remain_time = (((time.time()-time_start)/60)/(i+1))*(epochs-(i+1))

        time_avg.append(remain_time)

        mean_time = round(sum(time_avg[-5:])/5,3)

        print(f"Epoch [{epochs}/{i}] , total loss = {totalLoss/(len(test_loader.dataset)/test_loader.batch_size)} acc = {accuracy/len(test_loader.dataset)} , valid acc = {valid_acc} , estimate remain time : {mean_time}")



        if valid_acc < last_acc:


            trigger_times += 1

            print('trigger times:', trigger_times)

            if trigger_times >= patience:

                print(f"Early stopping! at {i+1} :)) ")

                return net

        else:

            print("reset trigger :)) ")

            trigger_times=0

        last_acc = valid_acc


        if best_acc < valid_acc:

            best_acc = valid_acc

            bestModel = net

            print("Saving best model")

            torch.save(bestModel.state_dict(), "model_ft/epoch"+str(i+1)+"_"+str(best_acc)+".pt")



    print("Complete training :)")




    return  bestModel



def run_pred(model,y_test,test_loader,device):

    print("start run predict...")

    model.eval()

    num  = torch.zeros(500 ,512)

    count =0

    with torch.no_grad():

        for idx ,(data,label) in  enumerate(test_loader):

            data = data.to(device)

            test_outputs = model(data)

            for i in range(len(test_outputs)):

                num[count] = test_outputs[i]

                count +=1

    acc = KNN(num , torch.tensor(y_test), batch_size=64)

    print(acc)


def run_pred(model,test_loader,device):

    print("start run predict...")

    model.eval()

    num  = torch.zeros(7294 ,512)

    count =0

    with torch.no_grad():

        for idx ,data in  enumerate(test_loader):

            data = data.to(device)

            test_outputs = model(data)

            for i in range(len(test_outputs)):

                num[count] = test_outputs[i]

                count +=1

    np.save("310704057.npy",num.cpu().detach().numpy())


def pred(model,y_test,test_loader,device):

    print("start run predict...")

    model.eval()

    num  = torch.zeros(500 ,512)

    count =0

    with torch.no_grad():

        for idx ,(data,label) in  enumerate(test_loader):

            data = data.to(device)

            test_outputs = model(data)

            for i in range(len(test_outputs)):

                num[count] = test_outputs[i]

                count +=1
    acc = KNN(num , torch.tensor(y_test), batch_size=32)

    print(acc)


