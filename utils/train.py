import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from A1.celeba_dataset import CelebaDataSet
from A1.model_a1 import ModelA1
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import config
from torch.utils.tensorboard import SummaryWriter


def train_A1():
    data = CelebaDataSet(config.CELEBA_IMG, config.CELEBA_LABELS)
    train_length = int(len(data)*0.7)
    train_data, val_data = random_split(data, lengths=[train_length, len(data)-train_length], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_data, batch_size=1,shuffle=True, num_workers=2)
    
    # device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")    
    device = torch.device("cpu")
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=3)
    model = ModelA1(num_classes = 1)
    model = model.to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5)
    
    writer = SummaryWriter('./path/to/log')
    num_epoch = 10
    for i in range(num_epoch):
        total_loss = 0.
        for idx, (img, label) in tqdm(enumerate(train_loader), total = len(train_loader)):
            img = img.to(device)
            label = label.to(device)
            out = model(img)
            loss = bce_loss(out.flatten(), label)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch:{}\t train loss:{}".format(i,
                                            total_loss/len(train_loader),
                                            ))
        writer.add_scalar("train/train_loss", total_loss/len(train_loader),i)
        
        total_acc = 0.
        total_f1 = 0.
        total_loss = 0.
        with torch.no_grad():
            for idx, (img, label) in tqdm(enumerate(val_loader), total = len(val_loader)):
                img = img.to(device)
                # label = label.to(device)
                out = model(img)
                # loss = bce_loss(out.flatten(), label)
                # total_loss += loss
                pred = out.sigmoid().detach().cpu().numpy() >=0.5
                # label = label.cpu().numpy()
                acc = accuracy_score(pred, label)
                f1 = f1_score(pred, label, zero_division=0)
                total_acc += acc
                total_f1 += f1
                
            print("epoch:{}\tval accuracy:{}\tval f1:{}".format(i,
                                                total_acc/len(val_loader),
                                                total_f1/len(val_loader)
                                                ))
            # writer.add_scalar("train/val_loss", total_loss/len(val_loader),i)
            writer.add_scalar("train/val_acc", total_acc/len(train_loader),i)
            writer.add_scalar("train/val_F1", total_f1/len(val_loader), i)
    writer.close()
    
def train_A2():
    pass

def train_B1():
    pass

def train_B2():
    pass