import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as tv
from PIL import Image
import decord
import numpy as np
from einops import reduce
from tqdm import tqdm
import gc
from cross_modality_fusion import CrossModality
from transformers import Blip2Processor, Blip2Model
from transformers import AutoImageProcessor, Dinov2Model
from datetime import datetime
import logging
import os
import random


random.seed(42)
torch.manual_seed(42)

# Dataset Class
class UCFDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, video_dir, subset, video_list_file, frames_per_clip=16):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.video_dir = video_dir
        self.subset = subset
        self.video_list_file = video_list_file
        self.video_list = []
        self.labels = []
        
        for i in [1, 2, 3]:
            file_path = f'{dataset_dir}/{video_list_file}{i}.txt'
            with open(file_path, 'r') as video_names_file:
                if self.subset == "train":
                    tempvideo_list, templabels = zip(*[line.strip().split() for line in video_names_file])
                    self.video_list.extend(tempvideo_list)
                    self.labels.extend(templabels)
                else:
                    self.video_list.extend([line.strip() for line in video_names_file])
                    self.labels.extend([None] * len(self.video_list))

        self.frames_per_clip = frames_per_clip

        self.transform = tv.transforms.Compose([
            # tv.transforms.GaussianBlur(9, sigma=(0.1, 2.0)),
            tv.transforms.Resize((224,224)),
            # tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2470, 0.2435, 0.2616]),
        ])

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        videoname = self.video_list[idx]
        video_path = f'{self.video_dir}/{videoname}'
        vid = decord.VideoReader(video_path, ctx=decord.cpu(0))
        nframes = len(vid)

        if nframes <= self.frames_per_clip:
            idxs = np.arange(self.frames_per_clip)
            idxs %= nframes
        else:
            idxs = np.linspace(0, nframes - 1, self.frames_per_clip).astype(int)

        imgs = [self.transform(Image.fromarray(vid[i].asnumpy())) for i in idxs]
        imgs = torch.stack(imgs,dim=1)

        if self.subset == "train":
            label = int(self.labels[idx]) - 1
        else:
            with open(f'{self.dataset_dir}/classInd.txt', 'r') as class_indices:
                class_map = {line.split()[1]: int(line.split()[0]) for line in class_indices}
            label = class_map[videoname.split('/')[0]]

        return imgs, label

# Model Class
class MLP(nn.Module):
    def __init__(self, dim, n_class, dino,blip):
        super().__init__()
        self.dino =dino
        self.dino.eval()
        self.mlp = nn.Linear(dim, n_class)
        self.h_dim=dim
    def forward(self, inp):
        with torch.no_grad():
            B,C,T,H,W = inp.shape
            inp = inp.transpose(1,2).reshape(B*T,C,H,W) 
            output = self.dino(inp).last_hidden_state

        output = output[:, 0, :]
        # print(output.shape)
        output = output.reshape(B,T,-1)
        # print(output.shape)
        output = output.mean(dim=-2)
        # print(output.shape)


        output = self.mlp(output) # b d -> b l
        # print(f"{output.shape=}")
        return output
# Parameters
dataset_dir = "UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/"
video_dir = "UCF101/UCF-101/"
frames_per_clip = 8
batch_size = 8
num_workers = 8
pin_memory = True
device = "cuda"
epochs = 25
lr = 2e-5
num_classes = 101

# Dataset and Dataloader
train_val_data = UCFDataset(dataset_dir, video_dir, "train", "trainlist0", frames_per_clip)
train_len = int(0.85 * len(train_val_data))

train_data, val_data = random_split(train_val_data, [train_len, len(train_val_data) - train_len])
print(len(train_data), len(val_data))

# test_data = UCFDataset(dataset_dir, video_dir, "test", "testlist0", frames_per_clip)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
# test_loader = DataLoader(test_data, batch_size=test_batch_size, num_workers=num_workers, pin_memory=pin_memory)

# Model, Loss, Optimizer, Scheduler
# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dino= Dinov2Model.from_pretrained('facebook/dinov2-base')
blip = Blip2Model.from_pretrained("Salesforce/blip2-flan-t5-xl")

blip.language_model = None

for param in dino.parameters():
    param.requires_grad = False

for param in blip.parameters():
    param.requires_grad = False



# dino.to(device)
# blip.to(device)

model = MLP(1024, num_classes,dino,blip)
model.to(device) 

for name,params in model.named_parameters():
    if(params.requires_grad):
        print(name)

loss_criterion = nn.CrossEntropyLoss()
T_max = epochs * len(train_loader)
optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

# Training and Validation Steps
def top_k_accuracy(output, target, topk=(1, 5)):
    """Computes the top-k accuracies, given the model output and true labels"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train_step(loader, epoch,scheduler):    
    model.train()
    total_loss = 0
    correct_top1_train = 0
    correct_top5_train = 0
    for batch_id, (video_data, labels) in tqdm(enumerate(loader),total=len(loader)):
        video_data, labels = video_data.to(device), labels.to(device)

        optimizer.zero_grad()
        prediction = model(video_data)
        # print(labels)
        # print(prediction.shape)
        # print(prediction)
        # for i,l in enumerate(labels):
        #     print(prediction[i][l])

        loss = loss_criterion(prediction, labels)
        total_loss += loss.item()
        # print(loss.item())
        # if batch_id%100==0:
        logging.info(f"[Train Epoch]: {epoch}, Batch: {batch_id}, Loss: {loss.item()},learning_rate: {optimizer.param_groups[0]["lr"]}")
        top1, top5 = top_k_accuracy(prediction, labels)
        correct_top1_train += top1.item()
        correct_top5_train += top5.item()
        

        loss.backward()
        optimizer.step()
        scheduler.step()

        del video_data, labels
        gc.collect()
##########################################################################################testing###############################
        # if batch_id==10:
        #     break
###########################################################################################################3


    top1_accuracy = correct_top1_train / len(loader)
    top5_accuracy = correct_top5_train / len(loader)    
    loss=total_loss/ len(loader.dataset)
    logging.info(f"[Training Epoch]: {epoch}, Top-1 Accuracy: {top1_accuracy:.4f}, Top-5 Accuracy: {top5_accuracy:.4f}, Loss: {loss:.4f},learning_rate: {optimizer.param_groups[0]["lr"]}")

    return total_loss

def val_step(loader, epoch):
    model.eval()
    total_loss = 0
    correct_top1_train = 0
    correct_top5_train = 0
    with torch.no_grad():
        for batch_id,(video_data, labels) in tqdm(enumerate(loader),total=len(loader)):
            video_data, labels = video_data.to(device), labels.to(device)
            prediction = model(video_data)
            loss = loss_criterion(prediction, labels)
            total_loss += loss.item()
            logging.info(f"[Val Epoch]: {epoch}, Loss: {loss.item()}")
            # corrects += (torch.argmax(prediction, dim=1) == labels).sum().item()
            top1, top5 = top_k_accuracy(prediction, labels)
            correct_top1_train += top1.item()
            correct_top5_train += top5.item()
##########################################################################################testing###############################
            # if batch_id==10:
            #     break
###########################################################################################################3
    
    top1_accuracy = correct_top1_train / len(loader)
    top5_accuracy = correct_top5_train / len(loader)
    loss=total_loss/ len(loader.dataset)
    # print(f"[Val Epoch]: {epoch}, Accuracy: {accuracy:.4f}, Loss: {total_loss:.4f}")
    logging.info(f"[Evaluation Epoch]: {epoch}, Top-1 Accuracy: {top1_accuracy:.4f}, Top-5 Accuracy: {top5_accuracy:.4f}, Loss: {loss:.4f}")
    return top1_accuracy

# Training Loop
if __name__=="__main__":
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Initialize logging configuration
    logging_dir="training_logs_correct"
    os.makedirs(logging_dir,exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,  # Set to DEBUG for more verbose logging
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # For console logging
            logging.FileHandler(f'{logging_dir}/cross_modality_dino_base_{current_time}.log', mode='w')  # 'w' to overwrite file each time the script runs
        ]
    )
    logging.info(f"Training started at {current_time}")

    new_acc=float("-inf")
    checkpoint_dir="checkpoints_corrected"
    os.makedirs(checkpoint_dir,exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_step(train_loader, epoch,scheduler)
        val_accuracy = val_step(val_loader, epoch)
        # scheduler.step()
        if(val_accuracy>new_acc):
            new_acc=val_accuracy
            logging.info(f"Saving model at epoch {epoch}")
            torch.save(model.state_dict(), os.path.join(checkpoint_dir,f"model_epoch_{current_time}_{epoch}.pth"))

            
