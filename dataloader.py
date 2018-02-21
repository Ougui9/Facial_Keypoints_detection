'''
Date:
author: Yilun Zhang
'''


import pandas as pd
import os
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import get_gt_map
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import torch
import torchvision.transforms as transforms
import numpy as np


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, trans_im, trans_label):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file, sep=" ", header=None,
                                           names=["#image path", "#x1","#x2","#x3","#x4","#x5","#y1","#y2","#y3"
                                               ,"#y4","#y5","#gender"," #smile", "#wearing glasses", "#head pose"])
        self.root_dir = root_dir
        self.trans_im = trans_im
        self.trans_label = trans_label
        # self.up_label = up_label

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.ix[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.ix[idx, 1:11].as_matrix().astype('float')
        # landmarks = landmarks.reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}
        sample = [image,landmarks]
        (H, W, C) = sample[0].shape

        if self.trans_im:
            sample[0] = self.trans_im(sample[0])

        if self.trans_label:
            sample[1] = F.upsample(Variable(torch.from_numpy(get_gt_map(sample[1].reshape([1,-1]), H, W)).float(),requires_grad=False), size=(40,40),mode='bilinear').data[0]

        # if self.up_label:
        #     sample[1] = self.up_label(Variable(sample[1].view(1,5,H,W),requires_grad=False)).data.view(5,40,40)
        return sample


