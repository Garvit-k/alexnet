# Created by Garvit Kothari at 03-07-2019
# Project Name : alexnet

# importing tensorflow and numpy
import tensorflow as tf
import numpy as np

# Convolutional Layer
def convLayer(x, height, width, strideX, strideY,
              featureNum, name, padding ="SAME", groups =1):
    channel = int(x.get_shape()[-1])

# Max Pooling Layer
def pool(x, height, width, strideX, strideY,
         name, padding = "SAME"):
    return tf.nn.max_pool(x, ksize = [1, height, width, 1],
                          strides = [1, strideX, strideY, 1],
                          padding = padding,
                          name = name)

# Response Normalization Layer
def RNL(x, R, alpha, beta, name = None, bias = 1.0):
    return tf.nn.local_response_normalization(x,
                                              depth_radius = R,
                                              alpha = alpha,
                                              beta = beta,
                                              bias = bias,
                                              name = name)
# dropout
def dropout(x, keepPro, name = None):
    return tf.nn.dropout(x, keepPro, name)




if __name__ == '__main__':
    pass
