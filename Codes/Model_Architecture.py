# DEFINE FUNCTIONS PARAMETERS, AND METRICES
import numpy as np
import keras.backend as K
import segmentation_models as sm
from keras.layers import (Input, Conv2D, Conv1D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, SpatialDropout2D,
                          Activation, Multiply, Add, DepthwiseConv2D, GlobalAveragePooling2D, Lambda, Reshape)
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Dice Loss Function
def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

loss = dice_loss

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

# Evaluation metrics - Precision, Recall, F1, IoU, MCC
metrics = [sm.metrics.Precision(threshold = 0.5), sm.metrics.Recall(threshold = 0.5), sm.metrics.FScore(threshold = 0.5, beta = 1), sm.metrics.IOUScore(threshold = 0.5), MCC]
metric_names = ['Loss', 'TP', 'FP', 'TN', 'FN', 'Precision', 'Recall', 'F1 Score', 'IoU Score', 'MCC']

# Convolution block
def conv_block(x, filters, dr=0.3):
    x = Conv2D(filters, 3, padding = 'same', kernel_initializer = 'glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, 3, padding = 'same', kernel_initializer = 'glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SpatialDropout2D(dr)(x)
    return x

def conv_block_(x, filters, dr=0.0):
    x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if dr > 0: x = SpatialDropout2D(dr)(x)

    return x

def ASPP_(x, filters):

    b1 = Conv2D(filters, 1, padding = "same", use_bias = False)(x)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)

    b2 = Conv2D(filters, 3, padding = "same", dilation_rate = 2, use_bias = False)(x)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)

    b3 = Conv2D(filters, 3, padding = "same", dilation_rate = 4, use_bias = False)(x)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)

    b4 = Conv2D(filters, 3, padding = "same", dilation_rate = 6, use_bias = False)(x)
    b4 = BatchNormalization()(b4)
    b4 = Activation("relu")(b4)

    x = concatenate([b1, b2, b3, b4])

    x = Conv2D(filters, 1, padding = "same", use_bias = False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def ECA(x, k_size = 3):
    c = x.shape[-1]

    y = GlobalAveragePooling2D()(x)
    y = Reshape((c, 1))(y)

    y = Conv1D(1, k_size, padding = "same", use_bias = False)(y)
    y = Activation("sigmoid")(y)

    y = Reshape((1, 1, c))(y)
    return Multiply()([x, y])

### TRIPLE ENCODER U-NET ###
def unet_triple(input_img, input_sar, input_dem, filters, lr, loss):

    # Inputs
    inp128 = Input(shape = input_img, name = "input_img")
    inp32_sar = Input(shape = input_sar, name = "input_sar")
    inp32_dem = Input(shape = input_dem, name = "input_dem")

    # Encoder (RGB-NIR 128×128)
    e1 = conv_block(inp128, filters)
    p1 = MaxPooling2D((2,2))(e1)

    e2 = conv_block(p1, filters*2)
    p2 = MaxPooling2D((2,2))(e2)

    e3 = conv_block(p2, filters*4)

    # Encoder (SAR – 32×32)
    e3_sar = conv_block(inp32_sar, filters*4, dr = 0.2)

    # Encoder (DEM – 32×32)
    e3_dem = conv_block(inp32_dem, filters*4, dr = 0.2)

    # Fusion 32×32
    fused_32 = concatenate([e3, e3_sar, e3_dem], axis = -1, name = "fusion_32")

    # Deep Encoder
    e4 = conv_block(fused_32, filters*8)

    p4 = MaxPooling2D((2,2))(e4)

    e5 = conv_block(p4, filters*16)
    p5 = MaxPooling2D((2,2))(e5)

    # Bottleneck
    b = conv_block(p5, filters*32, dr = 0.4)

    # Decoder
    u5 = UpSampling2D((2,2))(b)
    u5 = Conv2D(filters*16, 2, padding = "same", activation = "relu")(u5)
    d5 = conv_block(concatenate([e5, u5]), filters*16, dr = 0.2)

    u4 = UpSampling2D((2,2))(d5)
    u4 = Conv2D(filters*8, 2, padding = "same", activation = "relu")(u4)
    d4 = conv_block(concatenate([e4, u4]), filters*8, dr = 0.2)

    d3 = conv_block(concatenate([fused_32, d4]), filters*4, dr = 0.2)

    u2 = UpSampling2D((2,2))(d3)
    u2 = Conv2D(filters*2, 2, padding = "same", activation = "relu")(u2)
    d2 = conv_block(concatenate([e2, u2]), filters*2, dr = 0.2)

    u1 = UpSampling2D((2,2))(d2)
    u1 = Conv2D(filters, 2, padding = "same", activation = "relu")(u1)
    d1 = conv_block(concatenate([e1, u1]), filters, dr = 0.1)

    # Output
    out = Conv2D(1, 1, activation = "sigmoid")(d1)

    model = Model(inputs = [inp128, inp32_sar, inp32_dem], outputs = out)
    model.compile(optimizer = Adam(learning_rate = lr), loss = loss, metrics = metrics)

    return model

### TRIPLE U-NET++ ###
def unetpp_triple(input_img, input_sar, input_dem, filters, lr, loss):

    # Inputs
    inp128 = Input(shape = input_img, name = "input_img")
    inp32_sar = Input(shape = input_sar, name = "input_sar")
    inp32_dem = Input(shape = input_dem, name = "input_dem")

    # Encoder
    X0_0 = conv_block_(inp128, filters, dr = 0.0)
    X1_0 = conv_block_(MaxPooling2D(2)(X0_0), filters*2, dr = 0.0)
    X2_0_opt = conv_block_(MaxPooling2D(2)(X1_0), filters*4, dr = 0.1)

    # SAR & DEM encoders (32×32)
    X2_0_sar = conv_block_(inp32_sar, filters*4, dr = 0.0)
    X2_0_dem = conv_block_(inp32_dem, filters*4, dr = 0.0)

    # Fusion at 32×32
    X2_0 = concatenate([X2_0_opt, X2_0_sar, X2_0_dem], name = "fusion_32")
    X2_0 = conv_block_(X2_0, filters*4, dr = 0.2)

    X3_0 = conv_block_(MaxPooling2D(2)(X2_0), filters*8, dr = 0.3)
    X4_0 = conv_block_(MaxPooling2D(2)(X3_0), filters*16, dr = 0.3)

    # Decoder

    # Level 1
    X3_1 = conv_block_(concatenate([X3_0, UpSampling2D(2)(X4_0)]), filters*8, dr = 0.2)
    X2_1 = conv_block_(concatenate([X2_0, UpSampling2D(2)(X3_1)]), filters*4, dr = 0.1)
    X1_1 = conv_block_(concatenate([X1_0, UpSampling2D(2)(X2_1)]), filters*2, dr = 0.0)
    X0_1 = conv_block_(concatenate([X0_0, UpSampling2D(2)(X1_1)]), filters, dr = 0.0)

    # Level 2
    X2_2 = conv_block_(concatenate([X2_0, X2_1, UpSampling2D(2)(X3_1)]), filters*4, dr = 0.1)
    X1_2 = conv_block_(concatenate([X1_0, X1_1, UpSampling2D(2)(X2_2)]), filters*2, dr = 0.0)
    X0_2 = conv_block_(concatenate([X0_0, X0_1, UpSampling2D(2)(X1_2)]), filters, dr = 0.0)

    # Level 3
    X1_3 = conv_block_(concatenate([X1_0, X1_1, X1_2, UpSampling2D(2)(X2_2)]), filters*2, dr = 0.0)
    X0_3 = conv_block_(concatenate([X0_0, X0_1, X0_2, UpSampling2D(2)(X1_3)]), filters, dr = 0.0)

    # Final Output
    out = Conv2D(1, 1, activation = "sigmoid", name = "final")(X0_3)

    model = Model(inputs = [inp128, inp32_sar, inp32_dem], outputs = out)
    model.compile(optimizer = Adam(learning_rate = lr), loss = loss, metrics = metrics)

    return model

### TRIPLE U-NET++ w/ ECA ###
def unetpp_triple_eca(input_img, input_sar, input_dem, filters, lr, loss):

    # Inputs
    inp128 = Input(shape = input_img, name = "input_img")
    inp32_sar = Input(shape = input_sar, name = "input_sar")
    inp32_dem = Input(shape = input_dem, name = "input_dem")

    # Encoder
    X0_0 = conv_block_(inp128, filters, dr = 0.0)
    X1_0 = conv_block_(MaxPooling2D(2)(X0_0), filters*2, dr = 0.0)
    X2_0_opt = conv_block_(MaxPooling2D(2)(X1_0), filters*4, dr = 0.1)

    # SAR & DEM encoders (32×32)
    X2_0_sar = conv_block_(inp32_sar, filters*4, dr = 0.0)
    X2_0_dem = conv_block_(inp32_dem, filters*4, dr = 0.0)

    # Fusion at 32×32
    X2_0 = concatenate([X2_0_opt, X2_0_sar, X2_0_dem], name = "fusion_32")
    X2_0 = conv_block_(X2_0, filters*4, dr = 0.2)
    X2_0 = ECA(X2_0)  

    X3_0 = conv_block_(MaxPooling2D(2)(X2_0), filters*8, dr = 0.3)
    X4_0 = conv_block_(MaxPooling2D(2)(X3_0), filters*16, dr = 0.3)
    X4_0 = ECA(X4_0)

    # Decoder

    # Level 1
    X3_1 = conv_block_(concatenate([X3_0, UpSampling2D(2)(X4_0)]), filters*8, dr = 0.2)
    X2_1 = conv_block_(concatenate([X2_0, UpSampling2D(2)(X3_1)]), filters*4, dr = 0.1)
    X2_1 = ECA(X2_1)
    X1_1 = conv_block_(concatenate([X1_0, UpSampling2D(2)(X2_1)]), filters*2, dr = 0.0)
    X1_1 = ECA(X1_1)
    X0_1 = conv_block_(concatenate([X0_0, UpSampling2D(2)(X1_1)]), filters, dr = 0.0)

    # Level 2
    X2_2 = conv_block_(concatenate([X2_0, X2_1, UpSampling2D(2)(X3_1)]), filters*4, dr = 0.1)
    X1_2 = conv_block_(concatenate([X1_0, X1_1, UpSampling2D(2)(X2_2)]), filters*2, dr = 0.0)
    X0_2 = conv_block_(concatenate([X0_0, X0_1, UpSampling2D(2)(X1_2)]), filters, dr = 0.0)

    # Level 3
    X1_3 = conv_block_(concatenate([X1_0, X1_1, X1_2, UpSampling2D(2)(X2_2)]), filters*2, dr = 0.0)
    X0_3 = conv_block_(concatenate([X0_0, X0_1, X0_2, UpSampling2D(2)(X1_3)]), filters, dr = 0.0)

    # Final Output
    out = Conv2D(1, 1, activation = "sigmoid", name = "final")(X0_3)

    model = Model(inputs = [inp128, inp32_sar, inp32_dem], outputs = out)
    model.compile(optimizer = Adam(learning_rate = lr), loss = loss, metrics = metrics)

    return model

### TRIPLE U-NET++ w/ DEEP SUPERVISION & ECA ###
def unetpp_triple_deepsup_eca(input_img, input_sar, input_dem, filters, lr, loss):

    # Inputs
    inp128 = Input(shape = input_img, name = "input_img")
    inp32_sar = Input(shape = input_sar, name = "input_sar")
    inp32_dem = Input(shape = input_dem, name = "input_dem")

    # Encoder
    X0_0 = conv_block_(inp128, filters, dr = 0.0)
    X1_0 = conv_block_(MaxPooling2D(2)(X0_0), filters*2, dr = 0.0)
    X2_0_opt = conv_block_(MaxPooling2D(2)(X1_0), filters*4, dr = 0.1)

    # SAR & DEM encoders (32×32)
    X2_0_sar = conv_block_(inp32_sar, filters*4, dr = 0.0)
    X2_0_dem = conv_block_(inp32_dem, filters*4, dr = 0.0)

    # Fusion at 32×32
    X2_0 = concatenate([X2_0_opt, X2_0_sar, X2_0_dem], name = "fusion_32")
    X2_0 = conv_block_(X2_0, filters*4, dr = 0.2)
    X2_0 = ECA(X2_0)  

    X3_0 = conv_block_(MaxPooling2D(2)(X2_0), filters*8, dr = 0.3)
    X4_0 = conv_block_(MaxPooling2D(2)(X3_0), filters*16, dr = 0.3)
    X4_0 = ECA(X4_0)

    # Decoder

    # Level 1
    X3_1 = conv_block_(concatenate([X3_0, UpSampling2D(2)(X4_0)]), filters*8, dr = 0.2)
    X2_1 = conv_block_(concatenate([X2_0, UpSampling2D(2)(X3_1)]), filters*4, dr = 0.1)
    X2_1 = ECA(X2_1)
    X1_1 = conv_block_(concatenate([X1_0, UpSampling2D(2)(X2_1)]), filters*2, dr = 0.0)
    X1_1 = ECA(X1_1)
    X0_1 = conv_block_(concatenate([X0_0, UpSampling2D(2)(X1_1)]), filters, dr = 0.0)

    # Level 2
    X2_2 = conv_block_(concatenate([X2_0, X2_1, UpSampling2D(2)(X3_1)]), filters*4, dr = 0.1)
    X1_2 = conv_block_(concatenate([X1_0, X1_1, UpSampling2D(2)(X2_2)]), filters*2, dr = 0.0)
    X0_2 = conv_block_(concatenate([X0_0, X0_1, UpSampling2D(2)(X1_2)]), filters, dr = 0.0)

    # Level 3
    X1_3 = conv_block_(concatenate([X1_0, X1_1, X1_2, UpSampling2D(2)(X2_2)]), filters*2, dr = 0.0)
    X0_3 = conv_block_(concatenate([X0_0, X0_1, X0_2, UpSampling2D(2)(X1_3)]), filters, dr = 0.0)

    # Deep supervision outputs (all 128×128)
    out_1 = Conv2D(1, 1, activation="sigmoid", name="ds_1")(X0_1)
    out_2 = Conv2D(1, 1, activation="sigmoid", name="ds_2")(X0_2)
    out_f = Conv2D(1, 1, activation="sigmoid", name="final")(X0_3)

    model = Model(inputs = [inp128, inp32_sar, inp32_dem], outputs = [out_f, out_2, out_1])
    loss_weights = {"final": 1.0, "ds_2": 0.4, "ds_1": 0.2}
    model.compile(optimizer = Adam(learning_rate = lr), loss = {"final": loss, "ds_2": loss, "ds_1": loss}, loss_weights = loss_weights, metrics = {"final": metrics})

    return model

### TRIPLE U-NET++ w/ DEEP SUPERVISION, ECA, ASPP ###
def unetpp_triple_deepsup_eca_aspp(input_img, input_sar, input_dem, filters, lr, loss):

    # Inputs
    inp128 = Input(shape = input_img, name = "input_img")
    inp32_sar = Input(shape = input_sar, name = "input_sar")
    inp32_dem = Input(shape = input_dem, name = "input_dem")

    # Encoder
    X0_0 = conv_block_(inp128, filters, dr = 0.0)
    X1_0 = conv_block_(MaxPooling2D(2)(X0_0), filters*2, dr = 0.0)
    X2_0_opt = conv_block_(MaxPooling2D(2)(X1_0), filters*4, dr = 0.1)

    # SAR & DEM encoders (32×32)
    X2_0_sar = conv_block_(inp32_sar, filters*4, dr = 0.0)
    X2_0_dem = conv_block_(inp32_dem, filters*4, dr = 0.0)

    # Fusion at 32×32
    X2_0 = concatenate([X2_0_opt, X2_0_sar, X2_0_dem], name = "fusion_32")
    X2_0 = conv_block_(X2_0, filters*4, dr = 0.2)
    X2_0 = ECA(X2_0)  

    X3_0 = conv_block_(MaxPooling2D(2)(X2_0), filters*8, dr = 0.3)
    X4_0 = conv_block_(MaxPooling2D(2)(X3_0), filters*16, dr = 0.3)
    X4_0 = ECA(X4_0)
    X4_0 = ASPP_(X4_0, filters*16)

    # Decoder

    # Level 1
    X3_1 = conv_block_(concatenate([X3_0, UpSampling2D(2)(X4_0)]), filters*8, dr = 0.2)
    X2_1 = conv_block_(concatenate([X2_0, UpSampling2D(2)(X3_1)]), filters*4, dr = 0.1)
    X2_1 = ECA(X2_1)
    X1_1 = conv_block_(concatenate([X1_0, UpSampling2D(2)(X2_1)]), filters*2, dr = 0.0)
    X1_1 = ECA(X1_1)
    X0_1 = conv_block_(concatenate([X0_0, UpSampling2D(2)(X1_1)]), filters, dr = 0.0)

    # Level 2
    X2_2 = conv_block_(concatenate([X2_0, X2_1, UpSampling2D(2)(X3_1)]), filters*4, dr = 0.1)
    X1_2 = conv_block_(concatenate([X1_0, X1_1, UpSampling2D(2)(X2_2)]), filters*2, dr = 0.0)
    X0_2 = conv_block_(concatenate([X0_0, X0_1, UpSampling2D(2)(X1_2)]), filters, dr = 0.0)

    # Level 3
    X1_3 = conv_block_(concatenate([X1_0, X1_1, X1_2, UpSampling2D(2)(X2_2)]), filters*2, dr = 0.0)
    X0_3 = conv_block_(concatenate([X0_0, X0_1, X0_2, UpSampling2D(2)(X1_3)]), filters, dr = 0.0)

    # Deep supervision outputs (all 128×128)
    out_1 = Conv2D(1, 1, activation="sigmoid", name="ds_1")(X0_1)
    out_2 = Conv2D(1, 1, activation="sigmoid", name="ds_2")(X0_2)
    out_f = Conv2D(1, 1, activation="sigmoid", name="final")(X0_3)

    model = Model(inputs = [inp128, inp32_sar, inp32_dem], outputs = [out_f, out_2, out_1])
    loss_weights = {"final": 1.0, "ds_2": 0.4, "ds_1": 0.2}
    model.compile(optimizer = Adam(learning_rate = lr), loss = {"final": loss, "ds_2": loss, "ds_1": loss}, loss_weights = loss_weights, metrics = {"final": metrics})

    return model