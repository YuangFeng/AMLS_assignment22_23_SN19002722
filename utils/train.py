import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from A1.celeba_dataset import CelebaDataSet
from A1.model_a1 import ModelA1
from A2.model_a2 import Model_A2
from A2.lab2_lamdmarks import prepare_celeba_feature_labels
from B1.model_b1 import Model_B1  
from B2.model_b2 import Model_B2  
from B1.data_process import prepare_cartoon_data
from B2.data_process import prepare_cartoon_data2
  
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import config
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns



def evaluate(model, loader, device):
    label_list = []
    pred_list = []
    model.eval()
    with torch.no_grad():
        for idx, (img, label) in tqdm(enumerate(loader), total = len(loader)):
            img = img.to(device)
            out = model(img)
            pred = out.sigmoid().detach().cpu().numpy() >=0.5
            pred_list += list(pred)
            label_list += list(label)                
            
        acc = accuracy_score(label_list, pred_list)
        f1 = f1_score(label_list, pred_list, zero_division=0)
        cm = confusion_matrix(label_list, pred_list).ravel()
        
    return acc, f1, cm

def train_A1():
    print('#################### Run task A1 ###################')
    data = CelebaDataSet(config.CELEBA_IMG, config.CELEBA_LABELS)
    test_data = CelebaDataSet(config.CELEBA_IMG_TEST, config.CELEBA_TEST_LABELS)
    train_length = int(len(data)*0.8)
    print('train data:', train_length)
    print('val data:',  len(data)-train_length)
    print('test data:', len(test_data))
    train_data, val_data = random_split(data, lengths=[train_length, len(data)-train_length], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_data, batch_size=16,shuffle=True, num_workers=8) 
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=3)
    test_loader = DataLoader(test_data,  batch_size=1, shuffle=False, num_workers=3)
    
    device = torch.device("cpu")
    
    model = ModelA1(num_classes = 1)
    model = model.to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    writer = SummaryWriter('./runs/A1/tensorboard')
    num_epoch = 30
    best_acc = 0.0
    for i in range(num_epoch):
        model.train()
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
        
        acc, f1, cm = evaluate(model, val_loader, device)
        fig = sns.heatmap(cm, annot=True, fmt='d',cmap = 'Blues')
        heatmap = fig.get_figure()
        heatmap.savefig('A1/heatmap_A1.png', dpi = 400)
        if acc> best_acc:
            best_acc = acc
            torch.save(model, 'A1/best_model.pth')
            model = torch.load('A1/best_model.pth')
            print('save model at epoch {}'.format(i))
        print("epoch:{}\tval accuracy:{}\tval f1:{}".format(i,
                                            acc,
                                            f1
                                            ))
        writer.add_scalar("train/val_acc", acc,i)
        writer.add_scalar("train/val_F1", f1, i)
    writer.close()
    
    print('training finished!')
    model = torch.load('A1/best_model.pth')
    acc, f1, cm = evaluate(model, val_loader, device)
    
    fig = sns.heatmap(cm, annot=True, fmt='d',cmap = 'Blues')
    heatmap = fig.get_figure()
    heatmap.savefig('A1_heatmap', dpi = 400)
    
    cm = cm.ravel()
    print('####### testing results ##########')
    print('1.test acc:{}\n2.test f1:{}\n3.confusion matrix: tn [{}], fp [{}], fn [{}], tp [{}]'.format(acc, f1, cm[0], cm[1], cm[2], cm[3]))
    
def train_A2():
    print('#################### Run task A2 ###################')
    train_x, train_y =  prepare_celeba_feature_labels(config.CELEBA_IMG, config.CELEBA_LABELS, img_name_columns=1, labels_columns=3)
    test_x, test_y = prepare_celeba_feature_labels(config.CELEBA_IMG_TEST, config.CELEBA_TEST_LABELS, img_name_columns=1, labels_columns=3)
    
    model = Model_A2()
    # print(train_x)
    print('start training.....')
    model.train(train_x, train_y)
    print('end training.....')
    acc, f1, roc_data, cm = model.test(test_x, test_y)
    print('####### testing results ##########')
    print('1.test acc:{}\n2.test f1:{}\n3.confusion matrix: tn [{}], fp [{}], fn [{}], tp [{}]'.format(acc, f1, cm[0], cm[1], cm[2], cm[3]))

def train_B1():
    print('#################### Run task B1 ###################')
    train_x, train_y =  prepare_cartoon_data(config.CARTOON_IMG, config.CARTOON_LABELS, img_name_columns=3, labels_columns=2)
    test_x, test_y = prepare_cartoon_data(config.CARTOON_IMG_TEST, config.CARTOON_TEST_LABELS, img_name_columns=3, labels_columns=2)
    print('start training.....')
    model = Model_B1()
    model.train(train_x, train_y)
    print('end training.....')
    print('####### testing results ##########')
    acc, p_class, r_class, f_class = model.test(test_x, test_y)
    print('1.test acc:{}\n2.test f1:{}'.format(acc, f_class))
    

def train_B2():
    print('#################### Run task B2 ###################')
    train_x, train_y =  prepare_cartoon_data2(config.CARTOON_IMG, config.CARTOON_LABELS, img_name_columns=3, labels_columns=1, train=True)
    test_x, test_y = prepare_cartoon_data2(config.CARTOON_IMG_TEST, config.CARTOON_TEST_LABELS, img_name_columns=3, labels_columns=1, train = True)
    print('start training.....')
    model = Model_B2()
    model.train(train_x, train_y)
    print('end training.....')
    print('####### testing results ##########')
    acc, p_class, r_class, f_class = model.test(test_x, test_y)
    print('1.test acc:{}\n2.test f1:{}'.format(acc, f_class))
