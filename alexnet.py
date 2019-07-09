# Created by Garvit Kothari at 03-07-2019
# Project Name : alexnet

# importing tensorflow and numpy

import numpy as np
import tensorflow as tf

# Convolutional Layer
def convLayer(x, height, width, strideX, strideY,
              featureNum, name, padding="SAME", groups=1):
    channel = int(x.get_shape()[-1])
    conv = lambda a, b: tf.nn.conv2d(a, b,
                                     strides=[1, strideX, strideY, 1],
                                     padding=padding)
    with tf.get_variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[height, width, channel/groups, featureNum])
        b = tf.get_variable("b", shape=[featureNum])
        newx = tf.split(value=x, num_or_size_splits=groups, axis=3)
        neww = tf.split(value=w, num_or_size_splits=groups, axis=3)
        featureMap = [conv(t1,t2) for t1, t2 in zip(newx, neww)]
        mergeFeatureMap = tf.concat(axis=3, values=featureMap)
        out = tf.nn.bias_add(mergeFeatureMap, b)
        return tf.nn.relu(tf.reshape(out,
                          mergeFeatureMap.get_shape().as_list()),
                          name=scope.name)


# Max Pooling Layer
def pool(x, height, width, strideX, strideY,
         name, padding="SAME"):
    return tf.nn.max_pool(x, ksize=[1, height, width, 1],
                          strides=[1, strideX, strideY, 1],
                          padding=padding,
                          name=name)

# Response Normalization Layer
def RNL(x, R, alpha, beta, name=None, bias=1.0):
    return tf.nn.local_response_normalization(x,
                                              depth_radius=R,
                                              alpha=alpha,
                                              beta=beta,
                                              bias=bias,
                                              name=name)

# dropout
def dropout(x, keepPro, name=None):
    return tf.nn.dropout(x, keepPro, name)


# Fullyconnected Layer
def fullyConnectedLayer(x, inputDim, outputDim, reluFlag, name):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape=[inputDim, outputDim], dtype=float)
        b = tf.get_variable("b", [outputDim], dtype=float)
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out


class AlexNet(object):
    def __init__(self, x, keepPro, classNum, skip, modelPath = "mod_alexnet.npy"):
        self.X = x
        self.KEEPPRO = keepPro
        self.CLASSNUM = classNum
        self.SKIP = skip
        self.MODELPATH = modelPath
        self.buildModel()

    def buildModel(self):

        # 1st Conv Layer
        conv1 = convLayer(self.X, 11, 11, 4, 4, 96, "conv1", "VALID")
        lrn1 = RNL(conv1, 2, 2e-05, 0.75, "norm1")
        pool1 = pool(lrn1, 3, 3, 2, 2, "pool1", "VALID")

        # 2nd Conv Layer
        conv2 = convLayer(pool1, 5, 5, 1, 1, 256, "conv2", "VALID", 2)
        lrn2 = RNL(conv2, 2, 2e-05, 0.75, "lrn2")
        pool2 = pool(lrn2, 3, 3, 2, 2, "pool2", "VALID")

        # 3rd Conv Layer
        conv3 = convLayer(pool2, 3, 3, 1, 1, 384, "conv3")

        # 4th Conv Layer
        conv4 = convLayer(conv3, 3, 3, 1, 1, 384, "conv4", groups=2)

        # 5th Conv Layer
        conv5 = convLayer(conv4, 3, 3, 1, 1, 256, "conv5", groups=2)
        pool5 = pool(conv5, 3, 3, 2, 2, "pool5", "VALID")

        # Fully connected Layer input
        fcInput = tf.reshape(pool5, [-1, 256 * 6 * 6])

        # 1st FC Layer
        fc1 = fullyConnectedLayer(fcInput, 256*6*6, 4096, True, "fc1")
        dropout1 = dropout(fc1, self.KEEPPRO)

        # 2nd FC Layer
        fc2 = fullyConnectedLayer(dropout1, 4096, 4096, True, "fc2")
        dropout2 = dropout(fc2, self.KEEPPRO)

        # 3rd FC Layer final
        self.fc3 = fullyConnectedLayer(dropout2, 4096, self.CLASSNUM, True, "fc3")

    def loadModel(self, sess):
        wDict = np.load(self.MODELPATH, encoding="bytes").items()
        for name in wDict:
            if name not in self.SKIP:
                with tf.variable_scope(name, reuse=True):
                    for p in wDict[name]:
                        if len(p.shape) == 1:
                            sess.run(tf.get_variable('b', trainable=False).assign(p))
                        else:
                            sess.run(tf.get_variable('w', trainable=False).assign(p))
