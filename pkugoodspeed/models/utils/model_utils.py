import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy

'''
Mean IOU, which is the metrics for this training
'''
# Define IoU metric
'''
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2, y_true)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0) '''

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


'''
Define BCE_DICE_LOSS
'''
def dice_coef(y_true, y_pred):
    smooth = 1.e-5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    # return 0.5 * K.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)
    return -dice_coef(y_true, y_pred)

'''
DEFINE VAE_LOSS
'''
def vae_loss(y_true, y_pred):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    return K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)


loss_map = {
    'bin_cross': binary_crossentropy,
    'vae': vae_loss,
    'bce_dice': bce_dice_loss
}