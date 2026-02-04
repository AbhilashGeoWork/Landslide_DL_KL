# DEFINE FUNCTIONS PARAMETERS, AND METRICES
import numpy as np
import keras.backend as K
import segmentation_models as sm
from keras.layers import (Input, Conv2D, Conv1D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, SpatialDropout2D,
                          Activation, Multiply, Add, DepthwiseConv2D, GlobalAveragePooling2D, Lambda, Reshape)
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Tversky loss function
def tversky(y_true, y_pred, a=0.7):
    yt, yp = K.flatten(y_true), K.flatten(y_pred)
    tp = K.sum(yt * yp)
    fn = K.sum(yt * (1 - yp))
    fp = K.sum((1 - yt) * yp)
    return (tp + 1.0) / (tp + a * fn + (1 - a) * fp + 1.0)

def focal_tversky(y_true, y_pred, g=0.5):
    return K.pow((1.0 - tversky(y_true, y_pred)), g)

def tversky_loss(y_true, y_pred):
    return 1.0 - tversky(y_true, y_pred)

# MCC (Matthews Correlation Coefficient)
def MCC(y_true, y_pred):
    y_pred = tf.round(y_pred)
    y_true = tf.round(y_true)

    tp = tf.reduce_sum(y_true * y_pred)
    tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    numerator = tp * tn - fp * fn
    denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + tf.keras.backend.epsilon())