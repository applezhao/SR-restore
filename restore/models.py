import numpy as np
import tensorflow as tf

from keras.layers import Convolution2D, BatchNormalization, Activation, merge, UpSampling2D, Lambda, Input
from keras.models import Model
from keras import backend as K


##
## SR filter blocks
##
def conv_block(inputs, filters, kernel_size, strides=(1,1), padding="same", activation='relu'):
    x = Convolution2D(filters, kernel_size, strides=strides, padding=padding)(inputs)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)
    return x

def up_block(inputs, filters, kernel_size, strides=(1,1), scale=2, padding="same", activation="relu"):
    size = (scale,scale)
    x = UpSampling2D(size)(inputs)
    x = Convolution2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def res_block(inputs, filters=64):
    x = conv_block(inputs, filters, (3,3))
    x = conv_block(x, filters, (3,3), activation=False)
    return merge([x, inputs], mode='sum')


##
## RESTORE CNN
##

def restore_cnn_model(input_shape):
    inputs = Input(shape=input_shape)

    # 9-5-5 similar to SRCNN
    x = Convolution2D(64, (9, 9), activation='relu', padding='same', name='level1')(inputs)
    x = Convolution2D(32, (5, 5), activation='relu', padding='same', name='level2')(x)
    x = Convolution2D(4, (5, 5), activation='relu', padding='same', name='level3')(x)

    m = Model(inputs=inputs, outputs=x)
    return m

def restore_cnn_bn_model(input_shape):
    inputs = Input(shape=input_shape)

    # 9-5-5 w/ batch norm
    x = conv_block(inputs, 64, (9,9), activation='relu')
    x = conv_block(x, 32, (5,5), activation='relu')
    x = conv_block(x, 3, (5, 5), activation='relu')

    m = Model(inputs=inputs, outputs=x)
    return m


def restore_resnet_model(input_shape):
    inputs = Input(shape=input_shape)
    x = conv_block(inputs, 64, (9,9))
    for i in range(4): x = res_block(x)
    x = Convolution2D(3, (9,9), activation='relu', padding='same')(x)

    m = Model(inputs=inputs, outputs=x)
    return m
