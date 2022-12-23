import numpy as np
import matplotlib.pyplot as plt
import timm
import coat
import numpy as np
import torch.nn as nn
import torch
from torchvision import datasets, models, transforms
from typing import Any
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn import metrics
import gc
import os 
import torch
import torch.nn as nn
import sys
import pandas as pd
import wandb

from einops import rearrange
from einops.layers.torch import Rearrange
import PIL.Image as Image
import PIL as pil
import time
import einops
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.transforms import ImageOnlyTransform
import random_word as word

class restruct(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, img):
        """
        Args:
            img (Tensor): The stacked Image .
        Returns:
            Tensor: Restructured Image into 13 channels.
        """
    
        return einops.rearrange(torch.squeeze(img), 'h ( w c ) -> c h w ', w = 125, c=13)


def metric(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)
    return auc

def straightner(a):
    A = np.zeros((a[0].shape[0]*len(a)))
    start_index = 0
    end_index = 0
    for i in range(len(a)):
        start_index = i*a[0].shape[0]
        end_index = start_index+a[0].shape[0]
        A[start_index:end_index] = a[i]
    return A

def predictor(outputs):
    return np.argmax(outputs, axis = 1)



def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')
    

if __name__ == "__main__":


    """
    Total_No_epochs
    Present_Lr: Init Value when beginning the training
    Present_Wd: Init Value when beginning the training
    Run_Name: The name of the run
    Batch_Size
    """
    path = "/pscratch/sd/t/train130/dc250601/registry/"
    Total_No_epochs = int(sys.argv[1])
    Present_Lr = float(sys.argv[2])
    Present_Wd = float(sys.argv[3])
    Run_Name = sys.argv[4]
    Batch_Size = int(sys.argv[5])
    new_ = 1
    Present_Epoch = 0 
    if os.path.isfile(os.path.join(path,str(Run_Name)+".csv")):
        print("File Exits")
        df = pd.read_csv(os.path.join(path,str(Run_Name)+".csv"))
        Present_Lr = df["Present_Lr"][0]
        Present_Wd = df["Present_Wd"][0]
        Present_Epoch = df["Present_Epoch"][0]
        Run_Id = df["Run_Id"][0]
        Run_Name = df["Run_Name"][0]
        new_ = df["New"][0]
    else:
        df = pd.DataFrame()
        df["Total_No_epochs"] = [Total_No_epochs]
        df["Present_Lr"] = [Present_Lr]
        df["Present_Wd"]  = [Present_Wd]
        df["Present_Epoch"] = [Present_Epoch]
        df["Run_Id"] = [wandb.util.generate_id()]
        Run_Id = df["Run_Id"][0]
        df["New"] = [1]
        df["Run_Name"] = [Run_Name]
        df.to_csv(os.path.join(path,str(Run_Name)+".csv"))
        os.mkdir(f"/pscratch/sd/t/train130/dc250601/{Run_Name}")
        print("File Created")
        
    train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                restruct(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(60),])
    


    test_transform = transforms.Compose([
                                transforms.ToTensor(),
                                restruct()])
    

    dataset_train = datasets.ImageFolder("/dev/shm/Tau_Dataset/Train/",
                                        transform =train_transform,
                                        loader = pil_loader)
    
    dataset_test = datasets.ImageFolder("/dev/shm/Tau_Dataset/Test/",
                                        transform =test_transform,
                                        loader = pil_loader)
    

    dataloader_test = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=Batch_Size,
                                                shuffle=True,
                                                drop_last = True,
                                                num_workers=32,
                                                pin_memory = True)
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=Batch_Size,
                                                shuffle=True,
                                                drop_last = True,
                                                num_workers=32,
                                                pin_memory = True)
    
    
    image_size = (128,128)
    in_channels = 13
    num_blocks = [2, 2, 3, 5, 2]
    channels = [64, 96, 192, 384, 768]
    num_classes = 1
    model = coat.CoAtNet(image_size = image_size,
                            in_channels = in_channels,
                        num_blocks = num_blocks,
                        channels = channels,
                        num_classes = num_classes)
    
    
    
    if new_ == 0:

        checkpoint = torch.load(f"/pscratch/sd/t/train130/dc250601/{Run_Name}/model_Epoch_{Present_Epoch-1}.pt")
        model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.AdamW(model.parameters(), lr = Present_Lr, weight_decay = Present_Wd)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose = True,threshold = 0.0001,patience = 3, factor = 0.5)

    model = model.to("cuda")


    
    wandb.login(key="cb53927c12bd57a0d943d2dedf7881cfcdcc8f09")
    wandb.init(
        project = "Tau_Run2",
        name = Run_Name,
        id = Run_Id,
        resume = "allow"
    )


    scaler = torch.cuda.amp.GradScaler()
    #--------------------------
    wandb.watch(model, log_freq=50)
    #---------------------------
    w_intr = 50

    for epoch in range(Present_Epoch,Total_No_epochs,1):
        train_loss = 0
        val_loss = 0
        train_steps = 0
        test_steps = 0
        label_list = []
        outputs_list = []
        train_auc = 0
        test_auc = 0
        model.train()
        for image, label in tqdm(dataloader_train):
            image = image.to("cuda")
            label = label.to("cuda")
            with torch.no_grad():
                image = nn.functional.pad(image, (2,1,2,1))
            
            #optimizer.zero_grad()
            for param in model.parameters():
                param.grad = None

            with torch.cuda.amp.autocast():
                outputs = model(image)
                loss = criterion(outputs, label.float())
            label_list.append(label.detach().cpu().numpy())
            outputs_list.append(outputs.detach().cpu().numpy())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            train_steps += 1
            if train_steps%w_intr == 0:
                    wandb.log({"loss": loss.item()})
        with torch.no_grad():
            label_list = straightner(label_list)
            outputs_list = straightner(outputs_list)
            train_auc = metric(label_list, outputs_list) 




        #-------------------------------------------------------------------
        model.eval()
        label_list = []
        outputs_list = []
        with torch.no_grad():
            for image, label in tqdm(dataloader_test):
                image = image.to("cuda")
                image = nn.functional.pad(image, (2,1,2,1))
                label = label.to("cuda")
                outputs = model(image)
                loss = criterion(outputs, label.float())
                label_list.append(label.detach().cpu().numpy())
                outputs_list.append(outputs.detach().cpu().numpy())
                val_loss += loss.item()
                test_steps +=1
                if test_steps%w_intr == 0:
                    wandb.log({"val_loss": loss.item()})
            label_list = straightner(label_list)
            outputs_list = straightner(outputs_list)
            test_auc = metric(label_list, outputs_list)

        train_loss = train_loss/train_steps
        val_loss = val_loss/ test_steps
    #     hist_loss_train.append(train_loss)
    #     hist_loss_test.append(val_loss)
    #     hist_auc_train.append(train_auc)
    #     hist_auc_test.append(test_auc)

        print("----------------------------------------------------")
        print("Epoch No" , epoch)
        print("The Training loss of the epoch, ",train_loss)
        print("The Training AUC of the epoch,  %.3f"%train_auc)
        print("The validation loss of the epoch, ",val_loss)
        print("The validation AUC of the epoch, %.3f"%test_auc)
        print("----------------------------------------------------")
        PATH = f"/pscratch/sd/t/train130/dc250601/{Run_Name}/model_Epoch_{epoch}.pt"
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
                }, PATH)
        scheduler.step(test_auc)
        curr_lr = scheduler._last_lr[0]
        wandb.log({"Train_auc_epoch": train_auc,
                    "Epoch": epoch,
                    "Val_auc_epoch": test_auc,
                    "Train_loss_epoch": train_loss,
                    "Val_loss_epoch": val_loss,
                    "Lr": curr_lr}
                    )
        df = pd.read_csv(os.path.join(path,str(Run_Name)+".csv"))
        df["Present_Lr"][0] = curr_lr
        df["Present_Wd"][0]  = Present_Wd
        df["Present_Epoch"][0] = epoch + 1
        df["New"] = [0]
        df.to_csv(os.path.join(path,str(Run_Name)+".csv"))
        gc.collect()

    wandb.finish()
