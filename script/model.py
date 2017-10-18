# -*- coding:utf8 -*-

from __future__ import print_function
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, LSTM, GRU, Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import TimeDistributed, RepeatVector, Permute, Lambda, Bidirectional, Dropout
from keras.layers.merge import concatenate, add, dot, multiply
from keras.layers.normalization import BatchNormalization 
from keras import backend as K
from keras import regularizers 
from keras.layers import Activation 
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta, Adamax, Nadam
from keras.layers.advanced_activations import PReLU

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras import initializers
from keras.layers import merge


def tmp():
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=10))
    model.add(Dropout(0.25))
    model.add(RepeatVector(10))
    model.add(Conv1D(filters=32, kernel_size=2, strides=1, padding='valid', activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.25))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    # model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mse',
                  optimizer='nadam',
                  # optimizer='sgd',
                  metrics=['acc'])

    model.summary()
    return model

def baseline_model():
    x = Input(shape=(10,))

    hidden = Dense(128)(x)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)
    
    y = Dense(1, activation='relu')(hidden)
    model = Model(inputs=x, output=y)
    model.compile(loss='mse', optimizer='nadam', metrics=['acc'])

    model.summary()
    return model

def res_block(x, dense_hidden):

    hidden = Dense(dense_hidden)(x)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = concatenate([hidden, x])
    return hidden

def h_resnet_model():
    x = Input(shape=(10,))

    hidden = Dense(128)(x)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = res_block(hidden, 128)
    hidden = res_block(hidden, 128)
    hidden = res_block(hidden, 128)
    hidden = res_block(hidden, 64)
    hidden = res_block(hidden, 64)
    hidden = res_block(hidden, 64)
    hidden = res_block(hidden, 32)
    hidden = res_block(hidden, 32)
    hidden = res_block(hidden, 32)

    y = Dense(1,activation='relu')(hidden)

    model = Model(inputs=x, outputs=y)
    model.compile(loss='mse', optimizer='nadam', metrics=['acc'])
    model.summary()

    return model

def identity_block(x,k_dense):
    k1,k2,k3 = k_dense

    hidden = Dense(k1, use_bias=True, kernel_initializer='he_normal',bias_initializer='zeros')(x)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = Dense(k2, use_bias=True, kernel_initializer='he_normal',bias_initializer='zeros')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = Dense(k3, use_bias=True, kernel_initializer='he_normal',bias_initializer='zeros')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = merge([hidden,x], mode='concat')

    return hidden


def resnet_model():
    x = Input(shape=(10,))

    hidden = Dense(128)(x)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = identity_block(hidden,[128,128,128])
    hidden = identity_block(hidden,[128,64,64])
    hidden = identity_block(hidden,[64,32,32])
    hidden = identity_block(hidden,[32,16,16])
    hidden = identity_block(hidden,[8,8,4])

    y = Dense(1,activation='relu')(hidden)

    model = Model(inputs=x, outputs=y)
    model.compile(loss='mse', optimizer='nadam', metrics=['acc'])
    model.summary()

    return model


def q_model():
    
    x = Input(shape=(10,))
    hidden = Dense(128,use_bias=True,kernel_initializer='he_normal',
                bias_initializer='zeros')(x)
    hidden = Dropout(0.5)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = Dense(128,use_bias=True,kernel_initializer='he_normal',
                bias_initializer='zeros')(hidden)
    hidden = Dropout(0.5)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = Dense(64,use_bias=True,kernel_initializer='he_normal',
                bias_initializer='zeros')(hidden)
    hidden = Dropout(0.5)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = Dense(64,use_bias=True,kernel_initializer='he_normal',
                bias_initializer='zeros')(hidden)
    hidden = Dropout(0.5)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden) 

    hidden = Dense(32,use_bias=True,kernel_initializer='he_normal',
                bias_initializer='zeros')(hidden)
    hidden = Dropout(0.5)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = Dense(32,use_bias=True,kernel_initializer='he_normal',
                bias_initializer='zeros')(hidden)
    hidden = Dropout(0.5)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = Dense(16,use_bias=True,kernel_initializer='he_normal',
                bias_initializer='zeros')(hidden)
    hidden = Dropout(0.5)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = Dense(16,use_bias=True,kernel_initializer='he_normal',
                bias_initializer='zeros')(hidden)
    hidden = Dropout(0.5)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = Dense(8,use_bias=True,kernel_initializer='he_normal',
                bias_initializer='zeros')(hidden)
    hidden = Dropout(0.5)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = Dense(8,use_bias=True,kernel_initializer='he_normal',
                bias_initializer='zeros')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    y = Dense(1, activation='relu')(hidden)

    model = Model(inputs=x, outputs=y)

    #model.add(Dense(32, activation='relu', input_dim=10))
    #model.add(Dropout(0.5))
    #model.add(RepeatVector(10))
    #model.add(Conv1D(filters=32,kernel_size=3, strides=1, padding='valid', activation='relu'))
    #model.add(GlobalMaxPooling1D())
    #model.add(Dropout(0.5))
    #model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(1))
    model.compile(loss='mse', optimizer='nadam', metrics=['acc'])

    model.summary()
    return model

def conv_model():
    '''
    想办法实现卷积的网络
    '''
    x = Input(shape=(10,))
    hidden = Dense(32)(x)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)
    hidden = RepeatVector(10)(hidden)
    
    hidden = Conv1D(filters=32, kernel_size=2, strides=1, padding='valid')(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)
    
    hidden = GlobalAveragePooling1D()(hidden)
    
    hidden = Dense(32)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)
    
    # hidden = Dropout(0.25)(hidden)
    y = Dense(1, activation='relu')(hidden)

    model = Model(inputs=x, outputs=y)
    model.compile(loss='mse',
                  optimizer='nadam',
                  metrics=['acc'])
    model.summary()
    return model

def deeper_model():
    ''' 
    这个是最简单的神经网络模型
    '''
    x = Input(shape=(10,))
    hidden = Dense(128)(x)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = Dense(128)(x)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)
    
    hidden = Dense(128)(x)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)
    
    hidden = Dense(64)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = Dense(64)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)
    
    hidden = Dense(64)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = Dense(32)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = Dense(32)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    hidden = Dense(32)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)

    y = Dense(1, activation='relu')(hidden)
    
    model = Model(inputs=x, outputs=y)
    model.compile(loss='mse',
                  optimizer='nadam',
                  # optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['acc'])
    model.summary()
    return model


def first_model():
    ''' 
    这个是最简单的神经网络模型
    '''
    x = Input(shape=(10,))
    hidden = Dense(128)(x)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)
    
    hidden = Dense(64)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)
    
    hidden = Dense(32)(hidden)
    hidden = BatchNormalization()(hidden)
    hidden = Activation('tanh')(hidden)
    
    y = Dense(1, activation='relu')(hidden)
    # y = Dense(1, activation='sigmoid')(hidden)

    model = Model(inputs=x, outputs=y)
    model.compile(loss='mse',
                  optimizer='nadam',
                  # optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['acc'])
    model.summary()
    return model

if __name__ == '__main__':
    # first_model()
    conv_model()
    # model = tmp()
    # h_resnet_model()
    # SVG(model_to_dot(model).create(prog='dot', format='svg'))
