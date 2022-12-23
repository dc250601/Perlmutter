import timm
import torch
import numpy as np
import torch.nn as nn
import os
from tqdm.auto import tqdm
from sklearn.metrics import roc_curve
from sklearn import metrics
import gc
import sys
import pandas as pd
import wandb
import numpy as np
import matplotlib.pyplot as plt
import timm
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
import PIL.Image as Image
import einops
#Stage 1> A clean Swin transformer model will be trained trained
#Stage 2> The swin Transformer blocks are frozen and the new embeding layer is attached and trained
#The old embedding blocks are replaced by the CNNs.In this stage only the embeding layer is trained
#Stage 3> The entire model is unfrozen and trained

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


class restruct(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Args:
            img (Tensor): The stacked Image .
        Returns:
            Tensor: Restructured Image into 8 channels.
        """

        return einops.rearrange(torch.squeeze(img), 'h ( w c ) -> c h w ', w = 125, c=8)

def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

class Hybrid_embed(nn.Module):
    def __init__(self, feature_model, img_size, channels, efn_blocks, dims):
        super().__init__()


        self.feature_extractor = timm.create_model(feature_model,
                                                   features_only=True,
                                                   out_indices=[efn_blocks])


        self.feature_extractor.conv_stem = nn.Conv2d(channels,
                                       40,
                                       kernel_size=(3, 3),
                                       stride=(4, 4),
                                       padding=(1, 1),
                                       bias=False)

        with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = self.feature_extractor.training
                if training:
                    self.feature_extractor.eval()
                o = self.feature_extractor(torch.zeros(1, channels, img_size[0], img_size[1]))
                self.channel_output = o[0].shape[1]
                self.feature_extractor.train(training)

        self.embed_matcher = nn.Sequential(
            nn.Conv2d(self.channel_output, dims, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(dims, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.SiLU(inplace=True)
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.embed_matcher(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Hybrid_swin_effnet(nn.Module):
    def __init__(self, feature_model = "efficientnet_b3",img_size = (128,128), channels = 8, efn_blocks = 2, swin_blocks = 2, no_classes = 1):
        super().__init__()
        assert efn_blocks + swin_blocks == 4,f"The total no of blocks must be 4, instead {efn_blocks+swin_blocks} blocks provided "
        self.s1_flag = True
        self.s2_flag = True
        self.s3_flag = True
        self.swin_blocks = swin_blocks
        self.swin_backbone = timm.models.swin_transformer.SwinTransformer(img_size=128, 
                                                                          patch_size=4,
                                                                          in_chans=8,
                                                                          window_size=4,
                                                                          embed_dim=96,
                                                                          depths=(2, 2, 6, 2),
                                                                          num_heads=(3, 6, 12, 24))

        self.embeded_dim = self.swin_backbone.embed_dim * (2**(4 - self.swin_blocks))
        self.Hybrid_patch_embed = Hybrid_embed(feature_model = "efficientnet_b3",
                                                      img_size = (128,128),
                                                      channels = 8,
                                                      efn_blocks = 2,
                                                      dims = self.embeded_dim)
        self.swin_backbone.head = nn.Linear(self.swin_backbone.num_features, no_classes)


    def forward(self, image, stage):


        if stage == 2:
            #Attaching the new embeding layer

            self.swin_backbone.patch_embed = self.Hybrid_patch_embed

            for i in range((4- self.swin_blocks)):
                self.swin_backbone.layers[i] = nn.Identity()

            #Freezing the swin layers
            for layer in self.swin_backbone.layers:
                for para in layer.parameters():
                    para.requires_grad = False

           #Freezing the head
            for para in self.swin_backbone.head.parameters():
                    para.requires_grad = False

        if stage == 3:
            #Unfreezing the network
            for para in self.swin_backbone.parameters():
                para.requires_grad = True


        return self.swin_backbone(image).squeeze()


if __name__ == "__main__":

    """
    Total_No_epochs: The Total_No_epochs for stage 1
    Present_Lr: Init Value when beginning the training
    Present_Wd: Init Value when beginning the training
    Run_Name: The name of the run
    Batch_Size: Batch size to be used for training
    TEpoch2: The total number of epochs for stage 2
    LR2: The LR for stage 2
    WD2: The WD for stage 2

    """
    path = "/pscratch/sd/t/train130/dc250601/registry/"
    Total_No_epochs = int(sys.argv[1])
    Present_Lr = float(sys.argv[2])
    Present_Wd = float(sys.argv[3])
    Run_Name = sys.argv[4]
    Batch_Size = int(sys.argv[5])
    TEpoch2 = int(sys.argv[6])
    LR2 = float(sys.argv[7])
    WD2 = float(sys.argv[8])
    new_ = 1
    Present_Epoch = 0
    Current_Stage = 1

    if os.path.isfile(os.path.join(path,str(Run_Name)+".csv")):
        print("File Exits")
        df = pd.read_csv(os.path.join(path,str(Run_Name)+".csv"))
        Present_Lr = df["Present_Lr"][0]
        Present_Wd = df["Present_Wd"][0]
        Present_Epoch = df["Present_Epoch"][0]
        Run_Id = df["Run_Id"][0]
        Run_Name = df["Run_Name"][0]
        new_ = df["New"][0]
        Current_Stage = df["Current_Stage"][0]
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
        df["Current_Stage"] = [Current_Stage]

        df.to_csv(os.path.join(path,str(Run_Name)+".csv"))
        os.mkdir(f"/pscratch/sd/t/train130/dc250601/{Run_Name}")
        print("File Created")


    train_transform = transforms.Compose([
                                transforms.ToTensor(),
                                restruct(),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(20),])



    test_transform = transforms.Compose([
                                transforms.ToTensor(),
                                restruct()
                                ])


    dataset_train = datasets.ImageFolder("/dev/shm/Data_small/Train/",
                                        transform =train_transform,
                                        loader = pil_loader)

    dataset_test = datasets.ImageFolder("/dev/shm/Data_small/Test/",
                                        transform =test_transform,
                                        loader = pil_loader)


    criterion = nn.BCEWithLogitsLoss()
    model = Hybrid_swin_effnet()
    model = model.to("cuda")



    import wandb
    wandb.login(key="cb53927c12bd57a0d943d2dedf7881cfcdcc8f09")
    wandb.init( project = "Top_EFFSWIN",
                name = Run_Name,
                id = Run_Id,
                resume = "allow")
    

    sample = torch.randn(1, 8, 128, 128, device = "cuda")

    scaler = torch.cuda.amp.GradScaler()
    #--------------------------
    wandb.watch(model, log_freq=50)
    #---------------------------
    w_intr = 50

    if Current_Stage ==1:

        if new_==0:
            checkpoint = torch.load(f"/pscratch/sd/t/train130/dc250601/{Run_Name}/model_Epoch_{Present_Epoch-1}_stage1.pt")
            model.load_state_dict(checkpoint['model_state_dict'])

        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=Batch_Size, shuffle=True, drop_last = True, num_workers=32, pin_memory = True)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=Batch_Size, shuffle=True, drop_last = True, num_workers=32, pin_memory = True)

        print("Entering stage 1")
        #Stage 1 -----------------------------------------------------------------------------
        optimizer = torch.optim.AdamW(model.parameters(), lr = Present_Lr, weight_decay=Present_Wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose = True,threshold = 0.001,patience = 3, factor = 0.5)

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
                    outputs = model(image ,1)
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
                    label = label.to("cuda")
                    image = nn.functional.pad(image, (2,1,2,1))
                    outputs = model(image ,1)
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

            print("----------------------------------------------------")
            print("Epoch No" , epoch)
            print("The Training loss of the epoch, ",train_loss)
            print("The Training AUC of the epoch,  %.3f"%train_auc)
            print("The validation loss of the epoch, ",val_loss)
            print("The validation AUC of the epoch, %.3f"%test_auc)
            print("----------------------------------------------------")

            PATH = f"/pscratch/sd/t/train130/dc250601/{Run_Name}/model_Epoch_{epoch}_stage1.pt"
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                    }, PATH)
            scheduler.step(test_auc)
            curr_lr = scheduler._last_lr[0]

            df = pd.read_csv(os.path.join(path,str(Run_Name)+".csv"))
            df["Present_Lr"][0] = curr_lr
            df["Present_Wd"][0]  = Present_Wd
            df["Present_Epoch"][0] = epoch + 1
            df["New"] = [0]
            df.to_csv(os.path.join(path,str(Run_Name)+".csv"))


            wandb.log({"Train_auc_epoch": train_auc,
                    "Epoch": epoch,
                    "Val_auc_epoch": test_auc,
                    "Train_loss_epoch": train_loss,
                    "Val_loss_epoch": val_loss,
                    "Lr": curr_lr,
                    "Stage": 1
                    }
                    )
            gc.collect()


    if Current_Stage == 1:
        Current_Stage = 2
        df = pd.read_csv(os.path.join(path,str(Run_Name)+".csv"))
        df["Current_Stage"] = [2]
        df["Total_No_epochs"] = [TEpoch2]
        df["Present_Lr"] = [LR2]
        Present_Lr = LR2
        df["Present_Wd"]  = [WD2]
        Present_Wd = WD2
        df["Present_Epoch"] = [0]
        new_ = 1
        Present_Epoch = 0
        df["New"] = [new_]
        df.to_csv(os.path.join(path,str(Run_Name)+".csv"))



    if Current_Stage == 2:

        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=Batch_Size, shuffle=True, drop_last = True, num_workers=32, pin_memory = True)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=Batch_Size, shuffle=True, drop_last = True, num_workers=32, pin_memory = True)


        #Stage 2--------------------------------------------------------------------------------
        optimizer = torch.optim.AdamW(model.parameters(), lr = Present_Lr, weight_decay=Present_Wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', verbose = True,threshold = 0.001,patience = 3, factor = 0.5)


        if new_ == 1:
            checkpoint = torch.load(f"/pscratch/sd/t/train130/dc250601/{Run_Name}/model_Epoch_{Total_No_epochs-1}_stage1.pt")
            model.load_state_dict(checkpoint['model_state_dict'])

            with torch.no_grad():
                model(sample,2)
        else:
            with torch.no_grad():
                model(sample,2)
            checkpoint = torch.load(f"/pscratch/sd/t/train130/dc250601/{Run_Name}/model_Epoch_{Present_Epoch-1}_stage2.pt")
            model.load_state_dict(checkpoint['model_state_dict'])


        for epoch in range(Present_Epoch,TEpoch2,1):
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
                    outputs = model(image ,1)
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
                    label = label.to("cuda")
                    image = nn.functional.pad(image, (2,1,2,1))
                    outputs = model(image, 1)
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

            print("----------------------------------------------------")
            print("Epoch No" , epoch)
            print("The Training loss of the epoch, ",train_loss)
            print("The Training AUC of the epoch,  %.3f"%train_auc)
            print("The validation loss of the epoch, ",val_loss)
            print("The validation AUC of the epoch, %.3f"%test_auc)
            print("----------------------------------------------------")
            PATH = f"/pscratch/sd/t/train130/dc250601/{Run_Name}/model_Epoch_{epoch}_stage2.pt"
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                    }, PATH)
            scheduler.step(test_auc)
            curr_lr = scheduler._last_lr[0]

            df = pd.read_csv(os.path.join(path,str(Run_Name)+".csv"))
            df["Present_Lr"][0] = curr_lr
            df["Present_Wd"][0]  = Present_Wd
            df["Present_Epoch"][0] = epoch + 1
            df["New"] = [0]
            df["Current_Stage"] = [2]
            df.to_csv(os.path.join(path,str(Run_Name)+".csv"))

            wandb.log({"Train_auc_epoch": train_auc,
                    "Epoch": epoch,
                    "Val_auc_epoch": test_auc,
                    "Train_loss_epoch": train_loss,
                    "Val_loss_epoch": val_loss,
                    "Lr": curr_lr,
                    "Stage": 2
                    }
                    )
            gc.collect()

