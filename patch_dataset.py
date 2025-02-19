

import torch.utils.data
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from PIL import Image





class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, mode='train'):
        self.df = pd.read_excel(path)
        if mode == 'train':
            self.tf = transforms.Compose([
                transforms.RandomRotation(degrees=30),  # Random rotation between -30 and 30 degrees
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                transforms.ColorJitter(brightness=0.2),
                transforms.Resize((50, 50)),
                transforms.ToTensor()])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((50, 50)),
                transforms.ToTensor()])

        self.tf = transforms.Compose([
            transforms.Resize((50, 50)),
            transforms.ToTensor()])


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        id = item['id']
        label = item['label']
        path = item['path']
        loc = path.split('\\')[-1]
        img = np.load(path)
        input_tensor = self.tf(Image.fromarray(img))
        return np.array(input_tensor), label, id, loc




