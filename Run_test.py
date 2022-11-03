import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from Tool.Lookahead  import Lookahead

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


def train(net, trainLoader1,trainLoader2, optimizer, criterion,patience,scheduler, epochs ):

    testAccuracy  = 0
    trigger_times = 0
    last_loss     = 0

    best_loss     = 100000
    for i in range(0,epochs+1):

        net.train()

        totalLoss    = 0
        accuracy     = 0
        count        = 0


        for idx ,data in  enumerate(zip(trainLoader1,trainLoader2)):

            data_1 = data[0].cuda()
            data_2 = data[1].cuda()

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


        print(f"Epoch [{epochs}/{i}] , total loss = {totalLoss/(len(trainLoader1.dataset)/trainLoader1.batch_size)}")


        current_loss = totalLoss/(len(trainLoader1.dataset)/trainLoader1.batch_size)




        if current_loss > last_loss:

            trigger_times += 1

            print('trigger times:', trigger_times)

            if trigger_times >= patience:

                print(f"Early stopping! at {i+1} :)) ")

                return bestModel

        else:
            print('reset trigger : 0')

            trigger_times =0


        last_loss = current_loss



        if best_loss >= current_loss:

            best_loss = current_loss
            bestModel = net

            print("Saving best model")

            torch.save(bestModel.state_dict(), "model/epoch"+str(i+1)+"_"+str(best_loss)+".pt")
            bset_model_path = "model/epoch"+str(i+1)+"_"+str(best_loss)+".pt"


    print("Complete training :)")


    return bestModel

def train(net, trainLoader1,trainLoader2, optimizer, criterion,patience,scheduler, epochs ):

    testAccuracy  = 0
    trigger_times = 0
    last_loss     = 0

    best_loss     = 100000
    for i in range(0,epochs+1):

        net.train()

        totalLoss    = 0
        accuracy     = 0
        count        = 0


        for idx ,data in  enumerate(zip(trainLoader1,trainLoader2)):

            data_1 = data[0].cuda()
            data_2 = data[1].cuda()

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


        print(f"Epoch [{epochs}/{i}] , total loss = {totalLoss/(len(trainLoader1.dataset)/trainLoader1.batch_size)}")


        current_loss = totalLoss/(len(trainLoader1.dataset)/trainLoader1.batch_size)




        if current_loss > last_loss:

            trigger_times += 1

            print('trigger times:', trigger_times)

            if trigger_times >= patience:

                print(f"Early stopping! at {i+1} :)) ")

                return bestModel

        else:
            print('reset trigger : 0')

            trigger_times =0


        last_loss = current_loss



        if best_loss >= current_loss:

            best_loss = current_loss
            bestModel = net

            print("Saving best model")

            torch.save(bestModel.state_dict(), "model/epoch"+str(i+1)+"_"+str(best_loss)+".pt")
            bset_model_path = "model/epoch"+str(i+1)+"_"+str(best_loss)+".pt"


    print("Complete training :)")


    return best_model_path

def valid_test(model , test_loader,y_test):

    model.eval()

    num  = torch.zeros(500 ,512)

    count =0

    with torch.no_grad():

        for test_samples,_ in test_loader:

            test_samples = test_samples.cuda()

            test_outputs = model(test_samples)

            for i in range(len(test_outputs)):

                num[count] = test_outputs[i]

                count +=1

    acc = KNN(num , torch.tensor(y_test), batch_size=32)

    return acc

def fine_tune(net, path ,test_loader,optimizer,criterion,scheduler,patience,epochs):

    if path:
        net.load_state_dict(torch.load(path))

    trigger_times = 0
    last_acc      = 0
    best_acc      = 0

    for i in range(0,epochs+1):

        net.train()

        totalLoss    = 0
        accuracy     = 0
        count        = 0

        for idx ,(data,label) in  enumerate(test_loader):

            data  = data.cuda()
            label = label.cuda()
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


        valid_acc = valid_test(model = net , test_loader = test_loader , y_test = y_test)

        print(f"Epoch [{epochs}/{i}] , total loss = {totalLoss/(len(test_loader.dataset)/test_loader.batch_size)} acc = {accuracy/len(test_loader.dataset)} , valid acc = {valid_acc}")



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

            torch.save(bestModel.state_dict(), "model/epoch"+str(i+1)+"_"+str(best_acc)+".pt")
            bset_model_path = "model/epoch"+str(i+1)+"_"+str(best_acc)+".pt"




    print("Complete training :)")




    return bestModel,best_model_path



def run_pred(model,path,test_loader):

    if path:
        model.load_state_dict(torch.load(path))

    print("start run predict...")

    model.eval()


    num  = torch.zeros(500 ,512)

    count =0

    with torch.no_grad():

        for test_samples,label in test_loader:

            test_samples = test_samples.cuda()

            test_outputs = model(test_samples)

            for i in range(len(test_outputs)):

                num[count] = test_outputs[i]

                count +=1

    acc = KNN(num , torch.tensor(y_test), batch_size=64)

    print(acc)


# def run_final_output(model,,train_loader,path)

#     t = time.localtime()

#     # 依指定格式輸出
#     time_now = time.strftime("%m/%d/%Y-%H:%M:%S", t)

#     predictions = []
#     label = []

#     print("start run predict...")

#     model_ft.eval()
#     prediction2 = []


#     num  = torch.zeros(7249 ,512)

#     count =0

#     with torch.no_grad():

#         for test_samples in train_loader:
#             test_samples = test_samples.cuda()

#             test_outputs = model_ft(test_samples)

#             for i in range(len(test_outputs)):

#                 num[count] = test_outputs[i]

#                 count +=1

#     torch.save(num,f"final_output/final_{time_now}.npy")
