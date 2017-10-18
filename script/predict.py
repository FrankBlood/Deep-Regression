# -*- coding:utf8 -*-

import sys, os
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint
from data_loader import Data_Loader
from model import *

def predict(bst_model_path):
    data_loader = Data_Loader()
    val_data, val_label = data_loader.load(data_loader.validation_path)

    model = first_model()
    # model = conv_model()
    # model = tmp()

    y = model.predict(val_data)
    return y

def save(y, save_path):
    with open(save_path, 'w') as fw:
        for i in y:
            fw.write(str(i[0])+'\n')

if __name__ == "__main__":
    bst_model_path = sys.argv[1]
    y = predict(bst_model_path)
    save(y, sys.argv[2])

