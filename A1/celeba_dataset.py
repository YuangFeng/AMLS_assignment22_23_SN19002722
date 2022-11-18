import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
from os import path

class CelebaDataSet(Dataset):
    def __init__(self, imag_path, file_path):
        super().__init__()
        self.imag_path = imag_path
        self.file_path = file_path
        
        self.iamges, self.labels = self.load_data(file_path)
        self.transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
        ])
    
    def load_data(self, file_path):
        data = pd.read_csv(file_path, header='infer', sep="\t", usecols=[1,2,3])
        labels = dict()
        images = []
        for _, frame in data.iterrows():
           labels[frame['img_name']] = (frame['gender'] == 1)
           images.append(frame['img_name'])
        return images, labels
    
    def __getitem__(self, index):
        image_name = self.iamges[index]
        label1 = self.labels[image_name]
        img = Image.open( path.join(self.imag_path, image_name)).convert("RGB")
        img = self.transforms(img)
        return img, torch.tensor(label1, dtype=torch.float)
    
    def __len__(self):
        return len(self.iamges)    