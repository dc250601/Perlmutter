import numpy as np
import matplotlib.pyplot as plt
import coat
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

    print("Inside main--------------------------------------------------------")
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
                                                batch_size=256,
                                                shuffle=True,
                                                drop_last = True,
                                                num_workers=32,
                                                pin_memory = True)

    dataloader_train = torch.utils.data.DataLoader(dataset_test,
                                                batch_size=256,
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



    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.0005, weight_decay = 0.05)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose = True,threshold = 0.0001,patience = 3, factor = 0.5)

    model = model.to("cuda")



    scaler = torch.cuda.amp.GradScaler()

    w_intr = 50
    multiple = 2

    torch.cuda.cudart().cudaProfilerStart()
    for epoch in range(2):
        train_loss = 0
        val_loss = 0
        train_steps = 0
        test_steps = 0
        label_list = []
        outputs_list = []
        train_auc = 0
        test_auc = 0

        torch.cuda.nvtx.range_push("Epoch {}".format(epoch))
        torch.cuda.nvtx.range_push("Training")
        model.train()
        for image, label in tqdm(dataloader_train):
            torch.cuda.nvtx.range_push("Data_Transfer_to_cuda")
            image = image.to("cuda")
            label = label.to("cuda")
            with torch.no_grad():
                image = nn.functional.pad(image, (2,1,2,1))
            torch.cuda.nvtx.range_pop()
            #optimizer.zero_grad()

            for param in model.parameters():
                param.grad = None

            torch.cuda.nvtx.range_push("Forward Pass")
            with torch.cuda.amp.autocast():
                outputs = model(image)
                loss = criterion(outputs, label.float())
            torch.cuda.nvtx.range_pop()

            label_list.append(label.detach().cpu().numpy())
            outputs_list.append(outputs.detach().cpu().numpy())

            torch.cuda.nvtx.range_push("Backward pass")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.nvtx.range_pop()

            train_loss += loss.item()
            train_steps += 1
        
            if train_steps%(w_intr*multiple) == 0:
                break

        with torch.no_grad():
            label_list = straightner(label_list)
            outputs_list = straightner(outputs_list)
            train_auc = metric(label_list, outputs_list)


        torch.cuda.nvtx.range_pop()

        #-------------------------------------------------------------------
        torch.cuda.nvtx.range_push("Inference pass")
        model.eval()
        label_list = []
        outputs_list = []
        with torch.no_grad():
            for image, label in tqdm(dataloader_test):
                torch.cuda.nvtx.range_push("Data_Transfer_to_cuda")
                image = image.to("cuda")
                image = nn.functional.pad(image, (2,1,2,1))
                label = label.to("cuda")
                torch.cuda.nvtx.range_pop()

                torch.cuda.nvtx.range_push("Forward Pass")
                outputs = model(image)
                torch.cuda.nvtx.range_pop()

                loss = criterion(outputs, label.float())
                label_list.append(label.detach().cpu().numpy())
                outputs_list.append(outputs.detach().cpu().numpy())
                val_loss += loss.item()
                test_steps +=1
                
                if test_steps%(w_intr*multiple) == 0:
                    break
            label_list = straightner(label_list)
            outputs_list = straightner(outputs_list)
            test_auc = metric(label_list, outputs_list)

        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()
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

        scheduler.step(test_auc)
        curr_lr = scheduler._last_lr[0]
        
        gc.collect()
    torch.cuda.cudart().cudaProfilerStop()

