from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data
from PIL import Image
from gan import generator
from datasets import VOCDataSet
from transform import ReLabel, ToLabel
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
import tqdm
from Criterion import Criterion
import numpy as np
import os
from gycutils.gycaug import ColorAug,Random_horizontal_flip,Random_vertical_flip,Compose_imglabel,Random_crop,Random_rotation
input_transform = Compose([
    ColorAug(),
    ToTensor(),
    Normalize([.585, .256, .136], [.229, .124, .095]),
    ])
val_transform = Compose([
    ToTensor(),
    Normalize([.585, .256, .136], [.229, .124, .095]),
    ])

G=torch.nn.DataParallel(generator(n_filters=32)).cuda()

G.load_state_dict(torch.load("./pth/G.pth"))
G.eval()
files = []
for file_path in files:
    img = Image.open(file_path)
    img = val_transform(img)
    img = Variable( img ).cuda()
    prediction = G(img)
