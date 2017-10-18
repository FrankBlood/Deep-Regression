# -*- coding:utf8 -*-


import sys, os
from data_loader import Data_Loader
import numpy as np

data_loader = Data_Loader()
_, val_label = data_loader.load(data_loader.validation_path)

val_label = np.array(val_label)
target_label = (val_label*249+1)/10

with open('target.label', 'w') as fw:
    for label in target_label:
        fw.write(str(label)+'\n')

