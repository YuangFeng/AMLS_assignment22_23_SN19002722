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
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import config
from torch.utils.tensorboard import SummaryWriter


def train_A1():
    print('#################### Run task A1 ###################')
    data = CelebaDataSet(config.CELEBA_IMG, config.CELEBA_LABELS)
    train_length = int(len(data)*0.7)
    train_data, test_data = random_split(data, lengths=[train_length, len(data)-train_length], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_data, batch_size=16,shuffle=True, num_workers=8)    
    device = torch.device("cpu")
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=3)
    model = ModelA1(num_classes = 1)
    model = model.to(device)
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    writer = SummaryWriter('./path/to/log3')
    num_epoch = 30
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
        # scheduler_lr.step()
        print("epoch:{}\t train loss:{}".format(i,
                                            total_loss/len(train_loader),
                                            ))
        writer.add_scalar("train/train_loss", total_loss/len(train_loader),i)
        
        total_loss = 0.
        label_list = []
        pred_list = []
    
        model.eval()
        with torch.no_grad():
            for idx, (img, label) in tqdm(enumerate(test_loader), total = len(test_loader)):
                img = img.to(device)
                # label = label.to(device)
                out = model(img)
                # loss = bce_loss(out.flatten(), label)
                # total_loss += loss
                pred = out.sigmoid().detach().cpu().numpy() >=0.5
                pred_list += list(pred)
                label_list += list(label)                
                
            acc = accuracy_score(pred_list, label_list)
            f1 = f1_score(pred_list, label_list, zero_division=0)
            # print("epoch:{}\tval accuracy:{}\tval f1:{}".format(i,
            #                                     total_acc/len(val_loader),
            #                                     total_f1/len(val_loader)
            #                                     ))
            print("epoch:{}\tval accuracy:{}\tval f1:{}".format(i,
                                                acc,
                                                f1
                                                ))
            # writer.add_scalar("train/val_loss", total_loss/len(val_loader),i)
            writer.add_scalar("train/val_acc", acc,i)
            writer.add_scalar("train/val_F1", f1, i)
    writer.close()
    
def train_A2():
    print('#################### Run task A2 ###################')
    train_x, train_y =  prepare_celeba_feature_labels(config.CELEBA_IMG, config.CELEBA_LABELS, img_name_colunms=1, labels_colunms=3)
    test_x, test_y = prepare_celeba_feature_labels(config.CELEBA_IMG_TEST, config.CELEBA_TEST_LABELS, img_name_colunms=1, labels_colunms=3)
    
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
    train_x, train_y =  prepare_cartoon_data(config.CARTOON_IMG, config.CARTOON_LABELS, img_name_colunms=3, labels_colunms=2)
    test_x, test_y = prepare_cartoon_data(config.CARTOON_IMG_TEST, config.CARTOON_TEST_LABELS, img_name_colunms=3, labels_colunms=2)
    print('start training.....')
    model = Model_B1()
    model.train(train_x, train_y)
    print('end training.....')
    print('####### testing results ##########')
    acc, p_class, r_class, f_class = model.test(test_x, test_y)
    print('1.test acc:{}\n2.test f1:{}'.format(acc, f_class))
    

def train_B2():
    print('#################### Run task B2 ###################')
    train_x, train_y =  prepare_cartoon_data2(config.CARTOON_IMG, config.CARTOON_LABELS, img_name_colunms=3, labels_colunms=1)
    test_x, test_y = prepare_cartoon_data2(config.CARTOON_IMG_TEST, config.CARTOON_TEST_LABELS, img_name_colunms=3, labels_colunms=1)
    print('start training.....')
    model = Model_B2()
    model.train(train_x, train_y)
    print('end training.....')
    print('####### testing results ##########')
    acc, p_class, r_class, f_class = model.test(test_x, test_y)
    print('1.test acc:{}\n2.test f1:{}'.format(acc, f_class))
    pass
