import torch 
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import pickle
import random
import torch.nn.functional as F
import argparse
from torchvision import transforms
import copy
from RandAug import RandAugment

class FlexMatchDataset(Dataset):
    def __init__(self, dataset_config, mode = "train"):
        """
        transform : augmentation 종류 및 사용 여부 
        """
        self.mode = mode
        self.data_dir = dataset_config.data_dir
        self.num_labeled_data = dataset_config.num_labeled_data
        self.num_unlabeled_data = dataset_config.num_unlabeled_data
        self.data, self.label = self.load_data()
        self.transform_weak, self.transform_strong = self.get_transform()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        X, y = self.data[idx], self.label[idx]-1
        if self.mode != "unlabeled" :
            X = self.transform_weak(X)
        elif self.mode == "unlabeled":
            X_weak = self.transform_weak(X)
            X_strong = self.transform_strong(X)
        
        return (idx, X_weak, X_strong, y) if self.mode == "unlabeled" else (idx, X, y)

    def load_data(self):
        if self.mode == "train":
            labeled_data = self.__load_np_file( "train_X.pkl").reshape(-1, 3, 96, 96).transpose(0, 3, 2, 1)
            labeled_y = self.__load_np_file("train_y.pkl")
            labeled_data = labeled_data[:self.num_labeled_data]
            labeled_y = labeled_y[:self.num_labeled_data]
            return labeled_data, labeled_y

        elif self.mode =="unlabeled":
            unlabeled_data = self.__load_np_file("unlabeled_X.pkl").reshape(-1, 3, 96, 96).transpose(0, 3, 2, 1)
            unlabeled_y = self.__load_np_file("unlabeled_y.pkl")
            unlabeled_data = unlabeled_data[:self.num_unlabeled_data]
            unlabeled_y = unlabeled_y[:self.num_unlabeled_data]
            return unlabeled_data, unlabeled_y

        elif self.mode == "test":
            test_data = self.__load_np_file("test_X.pkl").reshape(-1, 3, 96, 96).transpose(0, 3, 2, 1)
            test_y = self.__load_np_file("test_y.pkl")
            return test_data, test_y

    def __load_np_file(self, path):
        path = os.path.join(self.data_dir, path)
        with open(path, "rb") as f:
            file = pickle.load(f)
        return file

    # Codes From https://github.com/TorchSSL/TorchSSL/blob/main/datasets/ssl_dataset.py
    def get_transform(self):
        mean = [x / 255 for x in [112.4, 109.1, 98.6]]
        std= [x / 255 for x in [68.4, 66.6, 68.5]]

        transform_weak, transform_strong = None, None
        if self.mode == "train" :
            transform_weak = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        elif self.mode == "test" :
            transform_weak = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        elif self.mode == "unlabeled":
            transform_weak = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
            transform_strong = copy.deepcopy(transform_weak)
            transform_strong.transforms.insert(1, RandAugment())

        return transform_weak, transform_strong 

def main():
    import matplotlib.pyplot as plt
    from torchvision.transforms import functional as F

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', type=str, default='debug_dataset.yaml')
    args, _ = parser.parse_known_args()
    dataset_args = OmegaConf.load(f'./{args.config_dir}')

    train_dataset = FlexMatchDataset(dataset_args)
    unlabeled_dataset = FlexMatchDataset(dataset_args, mode="unlabeled")
    test_dataset = FlexMatchDataset(dataset_args, mode="test")

    print("train size : ", len(train_dataset))
    print("unlabeled size : ", len(unlabeled_dataset))
    print("test size : ", len(test_dataset))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    unlabeled_dataloader = torch.utils.data.DataLoader(unlabeled_dataset, batch_size=1, shuffle=True, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

    for (idx_train, X_train, y_train) in train_dataloader:
        print("train : ", X_train.shape, y_train.shape)
        X_train = F.to_pil_image(X_train[0])
        plt.imshow(X_train)
        break
    
    for (idx_ulb, X_ulb_weak, X_ulb_strong, y_ulb) in unlabeled_dataloader:
        print("unlabeled : ", X_ulb_weak.shape, X_ulb_strong.shape, y_ulb.shape)
        plt.imshow(X_ulb_weak[0].permute(1, 2, 0))
        plt.imshow(X_ulb_strong[0].permute(1, 2, 0))
        break

    for (idx_test, X_test, y_test) in test_dataloader:
        print("test : ", X_test.shape, y_test.shape)
        plt.imshow(X_test[0].permute(1, 2, 0))
        break


if __name__ == '__main__':
    main()