# -*- coding:utf8 -*-

import sys, os
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from data_loader import Data_Loader
from model import *

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
num_threads = os.environ.get('OMP_NUM_THREADS')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads)
# config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def train():
    data_loader = Data_Loader()
    data, label = data_loader.new_load(data_loader.train_path)
    val_data, val_label = data_loader.new_load(data_loader.validation_path)
    test_data, test_label = data_loader.new_load(data_loader.test_path)

    # model = first_model()
    # model = q_model()
    model = resnet_model()
    #model = h_resnet_model()
    # model = deeper_model()
    #model = baseline_model()
    # model = tmp()

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    now_time = '_'.join(time.asctime(time.localtime(time.time())).split(' '))
    bst_model_path = '../models/' + 'qstd_resnet_model' + '_' + now_time + '.h5'
    print('bst_model_path:', bst_model_path)
    model_checkpoint = ModelCheckpoint(bst_model_path, monitor='val_loss', save_best_only=True, save_weights_only=True)

    tb_cb=TensorBoard(log_dir='./q_resnet_tensorboard/', histogram_freq=1, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

    model.fit(data, label, batch_size=32,
              epochs=30, shuffle=True,
              validation_data=[test_data, test_label],
              callbacks=[model_checkpoint, tb_cb])
              # callbacks=[early_stopping, model_checkpoint, tb_cb])
    
    if os.path.exists(bst_model_path):
        model.load_weights(bst_model_path) 
    
    test_loss, test_acc = model.evaluate(val_data, val_label)
    print(test_loss, test_acc)

if __name__ == "__main__":
    train()
