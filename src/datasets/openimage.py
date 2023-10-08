import anytree
import numpy as np
import os
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index
import csv

class OpenImg(Dataset):
    data_name = 'openImg'

    def __init__(self, root, split, subset, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.subset = subset
        self.transform = transform
        # if not check_exists(self.processed_folder):
        #     self.process()
        self.img, self.target = self.process(self.split)
        # self.target = self.target[self.subset]
        # self.classes_counts = make_classes_counts(self.target)
        # self.classes_to_labels, self.classes_size = load(os.path.join(self.processed_folder, 'meta.pt'))
        # self.classes_to_labels, self.classes_size = self.classes_to_labels[self.subset], self.classes_size[self.subset]
        self.classes_size = 596 # TODO: fix this

    def __getitem__(self, index):
        """
        return a dict:
        1. img: PIL.Image
        2. label: int
        """
        imgName, target = self.img[index][1], int(self.target[index])
        img = Image.open(os.path.join(self.root + "/train", imgName))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return {"img": img, self.subset: target}

        # img, target = Image.fromarray(self.img[index]), torch.tensor(self.target[index])
        # input = {'img': img, self.subset: target}
        # if self.transform is not None:
        #     input = self.transform(input)
        # return input

    def __len__(self):
        return len(self.img)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return self.root

    def process(self, split):
        if not check_exists(self.raw_folder):
            raise Exception(f"Dataset {self.data_name} not found.")
        return self.make_data(split)
        # save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        # save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        # save(meta, os.path.join(self.processed_folder, 'meta.pt'))
        # return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nSubset: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.subset, self.transform.__repr__())
        return fmt_str

    def make_data(self, split):
        """
        return:
        1. train_img: List[str]
        2. train_target: List[int]
        """
        imgs = []
        labels = []
        with open(f"{self.root}/client_data_mapping/{split}.csv") as f:
            reader = csv.reader(f)
            for row in reader:
                imgs.append(row[:-1])
                labels.append(int(row[1]))
        return imgs, labels
        # train_filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        # test_filenames = ['test_batch']
        # train_img, train_label = read_pickle_file(os.path.join(self.raw_folder, 'cifar-10-batches-py'), train_filenames)
        # test_img, test_label = read_pickle_file(os.path.join(self.raw_folder, 'cifar-10-batches-py'), test_filenames)
        # train_target, test_target = {'label': train_label}, {'label': test_label}
        # with open(os.path.join(self.raw_folder, 'cifar-10-batches-py', 'batches.meta'), 'rb') as f:
        #     data = pickle.load(f, encoding='latin1')
        #     classes = data['label_names']
        # classes_to_labels = {'label': anytree.Node('U', index=[])}
        # for c in classes:
        #     make_tree(classes_to_labels['label'], [c])
        # classes_size = {'label': make_flat_index(classes_to_labels['label'])}
        # return (train_img, train_target), (test_img, test_target)
