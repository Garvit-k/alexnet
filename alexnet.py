# Creation date : 03 - 07 - 2019
# author - Garvit Kothari

# importing tensorflow and numpy
import tensorflow as tf
import numpy as np

# Convolutional Layer
def convLayer(x, height, width, strideX, strideY, featureNum, name, padding ="SAME", groups =1):
    channel = int(x.get_shape()[-1])
