import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
from sklearn.metrics import jaccard_score


import numpy as np
import glob
import os
import wandb

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

# Initialize wandb with your API key and project name
# import getpass

# # Prompt for your API key
# api_key = getpass.getpass("Enter your wandb API key: ")

# # Initialize wandb with the API key
# wandb.login(key=api_key)

# Initialize wandb with your project name and hyperparameters
wandb.init(project="inspeklab", config={
    "learning_rate": 0.001,
    "batch_size_train": 12,
    "loss function": "DICE Loss"
    # Add more hyperparameters as needed
})

# ------- 1. define loss function --------

loss_function_name = 'muti_dice_loss'


# Define the Dice Loss function
def dice_loss(pred, target):
    smooth = 1.0
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice  # Return 1 - Dice to convert it to a loss


def muti_dice_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = dice_loss(d0, labels_v)
    loss1 = dice_loss(d1, labels_v)
    loss2 = dice_loss(d2, labels_v)
    loss3 = dice_loss(d3, labels_v)
    loss4 = dice_loss(d4, labels_v)
    loss5 = dice_loss(d5, labels_v)
    loss6 = dice_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
        loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(),
        loss5.data.item(), loss6.data.item()))

    return loss0, loss

def dice_score(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2.0 * intersection + smooth) / (union + smooth)

def iou_score(pred, target): 
    return jaccard_score(target.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy())
 

# ------- 2. set the directory of training dataset --------

model_name = 'u2net' #'u2netp'

checkpoint_dir = '/content/U-2-Net/saved_models'

data_dir_train = os.path.join(os.getcwd(), 'dataset' + os.sep +'train' + os.sep)
tra_image_dir = os.path.join('Image' + os.sep)
tra_label_dir = os.path.join('Mask' + os.sep) 

data_dir_val = os.path.join(os.getcwd(), 'dataset' + os.sep +'val' + os.sep)
val_image_dir = os.path.join('Image' + os.sep)
val_label_dir = os.path.join('Mask' + os.sep) 

image_ext = '.jpg' 
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

epoch_num = 200
batch_size_train = 12
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir_train + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
	img_name = img_path.split(os.sep)[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	tra_lbl_name_list.append(data_dir_train + tra_label_dir + imidx + label_ext)
     

val_img_name_list = glob.glob(data_dir_val + val_image_dir + '*' + image_ext)

val_lbl_name_list = []
for img_path in val_img_name_list:
	img_name = img_path.split(os.sep)[-1]

	aaa = img_name.split(".")
	bbb = aaa[0:-1]
	imidx = bbb[0]
	for i in range(1,len(bbb)):
		imidx = imidx + "." + bbb[i]

	val_lbl_name_list.append(data_dir_val + val_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

print("---")
print("valin images: ", len(val_img_name_list))
print("valin labels: ", len(val_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)
val_num = len(val_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

salobj_dataset_val = SalObjDataset(
    img_name_list=val_img_name_list,
    lbl_name_list=val_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))
salobj_dataloader_val = DataLoader(salobj_dataset_val, batch_size=batch_size_val, shuffle=False, num_workers=1)
	
# ------- 3. define model --------
# define the net
if(model_name=='u2net'):
    net = U2NET(3, 1)
elif(model_name=='u2netp'):
    net = U2NETP(3,1)

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 2000 # save the model every 2000 iterations

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_dice_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()

        # del temporary outputs and loss

        # Calculate Dice Score and IoU on GPU
        threshold = 0.5  # Adjust the threshold as needed

        # Apply threshold to model's prediction and target tensors
        pred_binary = (torch.sigmoid(d0).detach().cuda() >= threshold).float()
        labels_binary = (labels.cuda() >= threshold).float()

        # Calculate Dice Score and IoU with binary masks
        dice = dice_score(pred_binary, labels_binary)
        iou = iou_score(pred_binary, labels_binary)

        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        # Print statistics including Dice Score and IoU
        # print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, Dice Score: %3f, IoU: %3f" % (
        # epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num,
        # running_loss / ite_num4val, running_tar_loss / ite_num4val, dice, iou))

        wandb.log({
            "Training Loss": running_loss / ite_num4val,
            "Dice Score": dice,
            "IoU": iou
        }, step=ite_num)

        if ite_num % save_frq == 0:

            torch.save(net.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0


    # ------- 6. validation process --------
    net.eval()  # Set the model to evaluation mode

    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0

    with torch.no_grad():  # Disable gradient computation for validation
        for i, val_data in enumerate(salobj_dataloader_val):
            val_inputs, val_labels = val_data['image'], val_data['label']
            val_inputs = val_inputs.type(torch.FloatTensor).cuda() if torch.cuda.is_available() else val_inputs.type(torch.FloatTensor)
            val_labels = val_labels.type(torch.FloatTensor).cuda() if torch.cuda.is_available() else val_labels.type(torch.FloatTensor)

            # Forward pass for validation data
            val_d0, val_d1, val_d2, val_d3, val_d4, val_d5, val_d6 = net(val_inputs)
            val_loss2, val_loss = muti_dice_loss_fusion(val_d0, val_d1, val_d2, val_d3, val_d4, val_d5, val_d6, val_labels)

            # Calculate Dice Score and IoU for validation data
            threshold = 0.5
            val_pred_binary = (torch.sigmoid(val_d0) >= threshold).float()
            val_labels_binary = (val_labels >= threshold).float()
            val_dice += dice_score(val_pred_binary, val_labels_binary)
            val_iou += iou_score(val_pred_binary, val_labels_binary)
            val_loss += val_loss2

        val_loss /= len(salobj_dataloader_val)
        val_dice /= len(salobj_dataloader_val)
        val_iou /= len(salobj_dataloader_val)

        print("[epoch: %3d/%3d] val loss: %3f, Dice Score: %3f, IoU: %3f" % (epoch + 1, epoch_num, val_loss, val_dice, val_iou))

    wandb.log({
        "Validation Loss": val_loss,
        "Validation Dice Score": val_dice,
        "Validation IoU": val_iou
    })

checkpoint_name = f"{model_name}_{loss_function_name}_checkpoint_epoch_{epoch + 1}"
checkpoint_dir_path = os.path.join(checkpoint_dir, checkpoint_name)
os.makedirs(checkpoint_dir_path, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name,checkpoint_name+".pth")
torch.save(net.state_dict(), checkpoint_path)
print(f"Model checkpoint saved at {checkpoint_path}")

wandb.finish()
