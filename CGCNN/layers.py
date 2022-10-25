# import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow import zeros

tf.disable_eager_execution()
import numpy as np


def weightVariables(shape, name):
    initial = tf.truncated_normal(shape=shape, mean=0, stddev=1)#产生正态分布的随机数
    return tf.Variable(initial, name=name)


def chebyshevCoefficient(chebyshevOrder, inputNumber, outputNumber):
    chebyshevWeights = dict()
    for i in range(chebyshevOrder):
        initial = tf.truncated_normal(shape=[inputNumber, outputNumber], mean=0, stddev=1)#0.05
        chebyshevWeights['w_' + str(i)] = tf.Variable(initial)
    return chebyshevWeights

from utils import scaled_laplacian
def gcnLayer(inputPC, scaledLaplacian, pointNumber, inputFeatureN, outputFeatureN, chebyshev_order,rate):
    # scaledLaplacian = tf.nn.dropout(scaledLaplacian,keep_prob=0.5)
    # inputPC = tf.nn.dropout(inputPC,keep_prob=0.5)
    # w = glorot([inputFeatureN, outputFeatureN],name='123')
    # b = zeros([outputFeatureN],name='124')
    # p = tf.matmul(inputPC,w)
    # gcnOutput = tf.matmul(scaledLaplacian,p)
    # gcnOutput = gcnOutput + b
    # inputPC = tf.layers.batch_normalization(inputPC, momentum=0.9, training=True)
    # biasWeight = weightVariables([outputFeatureN,1,1,1], name='bias_w')
    biasWeight = weightVariables([outputFeatureN], name='bias_w')
    weight = weightVariables([inputFeatureN, outputFeatureN],name='123')

    chebyshevCoeff = chebyshevCoefficient(chebyshev_order, inputFeatureN, outputFeatureN)
    # support = tf.matmul(scaledLaplacian,inputPC)
    # # support = tf.nn.dropout(support, keep_prob=rate)
    # gcnOutput = tf.matmul(support,weight)
    # # gcnOutput = tf.layers.dense(support, outputFeatureN)
    # gcnOutput = gcnOutput + biasWeight
    chebyPoly = []
    cheby_K_Minus_1 = tf.matmul(scaledLaplacian, inputPC)
    cheby_K_Minus_2 = inputPC
    chebyPoly.append(cheby_K_Minus_2)
    chebyPoly.append(cheby_K_Minus_1)
    for i in range(2, chebyshev_order):
        chebyK = 2 * tf.matmul(scaledLaplacian, cheby_K_Minus_1) - cheby_K_Minus_2
        # chebyK = tf.matmul(scaledLaplacian, cheby_K_Minus_1)
        chebyPoly.append(chebyK)
        cheby_K_Minus_2 = cheby_K_Minus_1
        cheby_K_Minus_1 = chebyK
        #cheby_K_Minus_2, cheby_K_Minus_1 = cheby_K_Minus_1, chebyK
    chebyOutput = []
    for i in range(chebyshev_order):
        weights = chebyshevCoeff['w_' + str(i)]
        chebyPolyReshape = tf.reshape(chebyPoly[i], [-1, inputFeatureN])
        output = tf.matmul(chebyPolyReshape, weights)
        output = tf.reshape(output, [-1, pointNumber, outputFeatureN])
        chebyOutput.append(output)

    # chebyOutput = tf.convert_to_tensor(chebyOutput)
    # gcnOutput = chebyOutput + biasWeight
    #
    gcnOutput = tf.add_n(chebyOutput) + biasWeight
    # gcnOutput = tf.layers.batch_normalization(gcnOutput,epsilon=1e-05, momentum=0.5, axis=1, training=True)

    # axis = list(range(30 - 1))
    # mean, var = tf.nn.moments(gcnOutput,axis)
    # scale = tf.Variable(tf.ones(gcnOutput.shape[2]))
    # offset = tf.Variable(tf.zeros(gcnOutput.shape[2]))
    # variance_epsilon = 0.001
    # gcnOutput = tf.nn.batch_normalization(gcnOutput, mean, var, offset, scale, variance_epsilon)
    # gcnOutput = tf.nn.dropout(gcnOutput, keep_prob=0.5)
    # gcnOutput = tf.layers.dense(gcnOutput, outputFeatureN)
    # gcnOutput = tf.layers.batch_normalization(gcnOutput, training=True)
    gcnOutput = tf.nn.leaky_relu(gcnOutput)
    # w = glorot([outputFeatureN,outputFeatureN],name='12')
    # gcnOutput = tf.matmul(gcnOutput, w)
    # gcnOutput = gcnOutput + tf.layers.dense(inputPC, outputFeatureN)
    return gcnOutput

def glorot(shape, name=None):

    """Glorot & Bengio (AISTATS 2010) init."""

    init_range = np.sqrt(6.0/(shape[0]+shape[1]))

    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)

    return tf.Variable(initial, name=name)


def globalPooling(gcnOutput, featureNumber):
    #l2_max_pooling_pre = tf.reshape(gcnOutput, [-1, 1024, featureNumber, 1])
    #max_pooling_output_1=tf.nn.max_pool(l2_max_pooling_pre,ksize=[1,1024,1,1],strides=[1,1,1,1],padding='VALID')
    #max_pooling_output_1=tf.reshape(max_pooling_output_1,[-1,featureNumber])
    #mean, var = tf.nn.moments(gcnOutput, axes=[1])
    #poolingOutput = tf.concat([max_pooling_output_1, var], axis=1)

    mean, var = tf.nn.moments(gcnOutput, axes=[1]) #均值 方差
    max_f = tf.reduce_max(gcnOutput, axis=[1])
    poolingOutput = tf.concat([max_f, var], axis=1)
    #return max_f
    return poolingOutput

#fully connected layer without relu activation
def fullyConnected(features, inputFeatureN, outputFeatureN):
    weightFC = weightVariables([inputFeatureN, outputFeatureN], name='weight_fc')
    biasFC = weightVariables([outputFeatureN], name='bias_fc')
    outputFC = tf.matmul(features,weightFC)+biasFC
    outputFC = tf.nn.leaky_relu(outputFC)
    outputFC = tf.nn.dropout(outputFC, keep_prob=0.5)
    return outputFC
