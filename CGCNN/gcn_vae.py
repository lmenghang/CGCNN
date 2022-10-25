import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from layers import gcnLayer, fullyConnected, globalPooling

tf.disable_eager_execution()
# import torch
# from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

# Gaussian MLP as encoder

def gaussian_MLP_encoder(para,inputs,scaledLaplacian):

    # with tf.variable_scope("gaussian_MLP_encoder"):
    #
    #     # initializers
    #
    #     w_init = tf.contrib.layers.variance_scaling_initializer()
    #
    #     b_init = tf.constant_initializer(0.)
    #
    #     # 1st hidden layer
    #
    #     w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
    #
    #     b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
    #
    #     h0 = tf.matmul(x, w0) + b0
    #
    #     h0 = tf.nn.elu(h0)
    #
    #     h0 = tf.nn.dropout(h0, keep_prob)
    #
    #     # 2nd hidden layer
    #
    #     w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
    #
    #     b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
    #
    #     h1 = tf.matmul(h0, w1) + b1
    #
    #     h1 = tf.nn.tanh(h1)
    #
    #     h1 = tf.nn.dropout(h1, keep_prob)
    #
    #     # output layer
    #
    #     # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
    #
    #     wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2], initializer=w_init)
    #
    #     bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
    #
    #     gaussian_params = tf.matmul(h1, wo) + bo
    #     # The mean parameter is unconstrained
    #
    #     mean = gaussian_params[:, :n_output]
    #
    #     # The standard deviation must be positive. Parametrize with a softplus and
    #
    #     # add a small epsilon for numerical stability
    #
    #     stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])
    # encoder_h1 = tf.keras.layers.Dense(para.intermediate_dim[0], activation='relu')
    # encoder_h2 = tf.keras.layers.Dense(para.intermediate_dim[1], activation='relu')

    # encoder_h3 = Dense(intermediate_dim[2], activation='relu')
    # encoder_h4 = Dense(intermediate_dim[3], activation='relu')
    # encoder_mean = tf.keras.layers.Dense(para.latent_dim, activation='relu')
    # encoder_log_var = tf.keras.layers.Dense(para.latent_dim, activation='relu')
    # h1_encoded = encoder_h1(inputs)
    h1_encoded = fullyConnected(inputs,inputs.shape[2],para.intermediate_dim[0])
    # h1_encoded = tf.nn.relu(h1_encoded)
    h1_encoded = tf.nn.dropout(h1_encoded,keep_prob=para.keep_prob_1)
    h1_encoded = tf.layers.batch_normalization(h1_encoded)
    # h1_encoded = gcnLayer(inputs, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=inputs.shape[2],
    #               outputFeatureN=para.intermediate_dim[0],
    #               chebyshev_order=para.chebyshev_1_Order)
    # h2_encoded = encoder_h2(h1_encoded)
    h2_encoded = fullyConnected(h1_encoded,h1_encoded.shape[2],para.intermediate_dim[1])
    # h2_encoded = tf.nn.relu(h2_encoded)
    h2_encoded = tf.nn.dropout(h2_encoded,keep_prob=para.keep_prob_1)
    h2_encoded = tf.layers.batch_normalization(h2_encoded)

    # h2_encoded = gcnLayer(h1_encoded, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=h1_encoded.shape[2],
    #               outputFeatureN=para.intermediate_dim[1],
    #               chebyshev_order=para.chebyshev_1_Order)

    # z_mean = encoder_mean(h2_encoded)
    # z_log_var = encoder_log_var(h2_encoded)
    # print('h2_encoded.shape',h2_encoded.shape)
    z_mean = gcnLayer2(h2_encoded, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=h2_encoded.shape[2],
                  outputFeatureN=para.latent_dim,chebyshev_order=para.chebyshev_1_Order)
    z_mean = tf.nn.dropout(z_mean, keep_prob=para.keep_prob_1)
    z_log_var = gcnLayer2(h2_encoded, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=h2_encoded.shape[2],
                     outputFeatureN=para.latent_dim,chebyshev_order=para.chebyshev_1_Order)
    z_log_var = tf.nn.dropout(z_log_var,keep_prob=para.keep_prob_1)
    # z_log_var = tf.exp(z_log_var)
    return z_mean, z_log_var



# Bernoulli MLP as decoder

def bernoulli_MLP_decoder(para,scaledLaplacian,z, intermediate_dim):

    # with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
    #     # initializers
    #     w_init = tf.contrib.layers.variance_scaling_initializer()
    #
    #     b_init = tf.constant_initializer(0.)
    #
    #     # 1st hidden layer
    #     w0 = tf.get_variable('w0', [z.get_shape()[2], intermediate_dim[0]], initializer=w_init)
    #
    #     b0 = tf.get_variable('b0', [intermediate_dim[0]], initializer=b_init)
    #
    #     h0 = tf.matmul(z, w0) + b0
    #
    #     h0 = tf.nn.tanh(h0)
    #
    #     h0 = tf.nn.dropout(h0, keep_prob)
    #
    #     # 2nd hidden layer
    #     w1 = tf.get_variable('w1', [h0.get_shape()[2], intermediate_dim[1]], initializer=w_init)
    #
    #     b1 = tf.get_variable('b1', [intermediate_dim[1]], initializer=b_init)
    #
    #     h1 = tf.matmul(h0, w1) + b1
    #
    #     h1 = tf.nn.elu(h1)
    #
    #     h1 = tf.nn.dropout(h1, keep_prob)
    #
    #     # output layer-mean
    #     wo = tf.get_variable('wo', [h1.get_shape()[2], n_output], initializer=w_init)
    #
    #     bo = tf.get_variable('bo', [n_output], initializer=b_init)
    #
    #     y = tf.sigmoid(tf.matmul(h1, wo) + bo)
    # decoder_h1 = tf.keras.layers.Dense(intermediate_dim[1], activation='relu')
    # decoder_h2 = tf.keras.layers.Dense(intermediate_dim[0], activation='relu')
    # decoder_h1 = gcnLayer(z, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=z.shape[2],
    #               outputFeatureN=para.intermediate_dim[1],
    #               chebyshev_order=para.chebyshev_1_Order)
    # decoder_h2 = gcnLayer(decoder_h1, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=decoder_h1.shape[2],
    #               outputFeatureN=para.intermediate_dim[0],
    #               chebyshev_order=para.chebyshev_1_Order)
    # decoder_h3 = Dense(intermediate_dim[0], activation='relu')

    # decoder_mean = tf.keras.layers.Dense(z.shape[1], activation='tanh')
    h1_decoded = gcnLayer(z, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=z.shape[2],
                  outputFeatureN=intermediate_dim[1],
                  chebyshev_order=para.chebyshev_1_Order)
    h2_decoded = gcnLayer(h1_decoded, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=h1_decoded.shape[2],
                  outputFeatureN=intermediate_dim[0],
                  chebyshev_order=para.chebyshev_1_Order)
    # h3_decoded = decoder_h3(h2_decoded)
    # outputs = decoder_mean(h2_decoded)
    # print('h2_decoded',h2_decoded.shape)
    # scaledLaplacian = tf.concat([scaledLaplacian,scaledLaplacian],axis=0)
    outputs = gcnLayer(h2_decoded, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=h2_decoded.shape[2],
                  outputFeatureN=para.gcn_1_filter_n,
                  chebyshev_order=para.chebyshev_2_Order)
    # print('outputs.shape',outputs.shape)
    outputs = globalPooling(outputs, featureNumber=para.intermediate_dim[0])
    # inputshape = para.latent_dim * 2
    # print('y',outputs.shape)
    fc_layer_1 = fullyConnected(outputs, inputFeatureN=outputs.shape[1], outputFeatureN=para.fc_1_n)
    fc_layer_1 = tf.nn.tanh(fc_layer_1)
    fc_layer_1 = tf.nn.dropout(fc_layer_1, keep_prob=para.keep_prob_2)
    outputs = fullyConnected(fc_layer_1, inputFeatureN=fc_layer_1.shape[1], outputFeatureN=para.outputClassN)

    return outputs



# Gateway
from layers import weightVariables
from sklearn import preprocessing
# from tensorflow.contrib.layers import batch_norm
from sklearn import svm
def autoencoder(para):
    inputPC = tf.placeholder(tf.float32, [None, para.pointNumber, para.inputFeatureN])
    inputGraph = tf.placeholder(tf.float32, [None, para.pointNumber* para.pointNumber])
    outputLabel = tf.placeholder(tf.float32, [None, para.outputClassN])

    scaledLaplacian = tf.reshape(inputGraph, [-1, para.pointNumber, para.pointNumber])
    # scaledLaplacian = scaledLaplacian

    weights = tf.placeholder(tf.float32, [None])
    lr = tf.placeholder(tf.float32)
    keep_prob_1 = tf.placeholder(tf.float32)
    keep_prob_2 = tf.placeholder(tf.float32)
    # inputPC = tf.reshape(inputPC, [62,-1])
    # inputPC = tf.layers.dense(inputPC,128)
    # gcn layer 1
    # inputPC = tf.layers.batch_normalization(inputPC,epsilon=1e-05, momentum=0.1, axis=1, training=True)

    #gcn layer
    gcn_1 = gcnLayer(inputPC, scaledLaplacian, pointNumber=para.pointNumber, inputFeatureN=para.inputFeatureN,
                     outputFeatureN=para.gcn_1_filter_n,
                     chebyshev_order=para.chebyshev_1_Order,rate=keep_prob_2)
    gcn_1_output = tf.nn.dropout(gcn_1, keep_prob=keep_prob_1)
    print('gcn_output:', gcn_1_output)

    # # separable convolution
    # conv = tf.layers.separable_conv1d(gcn_1_output, filters=8, kernel_size=16, strides=1,data_format='channels_first',padding='same')
    # conv = tf.nn.elu(conv)
    # print('depthwise-separable-conv_output', conv.shape)
    # conv = tf.nn.dropout(conv, keep_prob=keep_prob_2)
    # conv = tf.layers.average_pooling1d(conv, 8, strides=8,data_format='channels_first')
    # print('ave_pooling_output', conv.shape)

    # #cnn
    # conv = tf.layers.conv1d(gcn_1_output, filters=8,kernel_size=16,strides=1,padding='valid',data_format='channels_first',)


    # separable convolution
    conv = tf.expand_dims(gcn_1_output,3)
    print('conv:', conv)
    conv = tf.layers.separable_conv2d(conv, filters=2, kernel_size=(1,4), strides=(1,1), padding='same')
    # conv = tf.layers.separable_conv1d(gcn_1_output, filters=8, kernel_size=16, strides=1,data_format='channels_first',padding='same')
    conv = tf.nn.elu(conv)
    print('depthwise-separable-conv_output', conv.shape)
    conv = tf.nn.dropout(conv, keep_prob=keep_prob_2)
    conv = tf.layers.average_pooling2d(conv, (1, 2), strides=(1, 2))
    # conv = tf.layers.average_pooling1d(conv, 8, strides=8,data_format='channels_first')
    print('ave_pooling_output', conv.shape)


    conv = tf.nn.dropout(conv, keep_prob=keep_prob_2)
    # print('conv.shape', conv.shape)

    conv = tf.reshape(conv, [-1, conv.shape[1]*conv.shape[2]*conv.shape[3]])
    # conv = tf.reshape(conv, [-1, conv.shape[1]*conv.shape[2]])
    # conv = tf.concat([conv, gcn], axis=1)

    y = tf.nn.dropout(conv, keep_prob=keep_prob_2)
    print('y.shape', y.shape)


    # y = tf.concat([gcn_2_output, conv], axis=2)
    # y = tf.concat([gcn_2_output,z],axis=2)
    # y = globalPooling(conv, featureNumber=conv.shape[2])
    # print('gcn_1_pooling.shape',gcn_1_pooling.shape)
    # y = tf.layers.batch_normalization(y)
    # fc_layer_1 = fullyConnected(y, inputFeatureN=y.shape[1], outputFeatureN=para.fc_1_n)
    # fc_layer_1 = tf.nn.relu(fc_layer_1)
    # fc_layer_1 = tf.nn.dropout(fc_layer_1, keep_prob=keep_prob_2)
    # fully connected layer 2

    # y = tf.layers.batch_normalization(y,training=True)
    # y = tf.nn.leaky_relu(y)
    # y = tf.nn.dropout(y,keep_prob=keep_prob_2)
    # y = tf.layers.dense(y, para.fc_1_n)
    # y = tf.keras.layers.Dense(para.fc_1_n, activation='s')(y)
    # y = tf.nn.relu(y)
    # y = tf.keras.layers.Dense(para.outputClassN, activation='softmax')(y)
    # y = tf.contrib.layers.fully_connected(y, para.fc_1_n, scope='hidden1', normalizer_fn=batch_norm, normalizer_params=bn_params)
    # y = tf.nn.dropout(y,keep_prob=keep_prob_2)
    # # y = tf.layers.batch_normalization(y, training=True)
    # y = tf.layers.batch_normalization(y,epsilon=1e-05, momentum=0.1, axis=1, training=True)
    # y = tf.nn.relu(y)
    # y = tf.nn.dropout(y,keep_prob=keep_prob_2)
    # y = tf.layers.batch_normalization(y, training=True)
    # y = tf.layers.dense(y, para.fc_1_n)
    y = tf.layers.dense(y, para.outputClassN)
    # y = fullyConnected(y, inputFeatureN=y.shape[1], outputFeatureN=para.fc_1_n)
    # y = tf.nn.leaky_relu(y, alpha=0.01)
    # y = fullyConnected(y, inputFeatureN=y.shape[1], outputFeatureN=para.outputClassN)
    # print('fc_layer_2.shape',fc_layer_2.shape)

    # y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

    # y = tf.layers.batch_normalization(y)
    # y = preprocessing.scale(y)
    # y = tf.nn.softmax(y)
    # y -= 0.5 * (y - 1. / tf.cast(y.shape[-1], y.dtype))
    predictsoftmax = tf.nn.softmax(y,1)
    predictLabels = tf.argmax(predictsoftmax, axis=1)


    # loss
    # marginal_likelihood = tf.reduce_sum(inputPC * tf.log(y) + (1 - inputPC) * tf.log(1 - y), 1)
    # KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

    # KL_divergence = 0.5 * tf.reduce_sum(tf.exp(sigma) + tf.square(mu) - 1. - sigma,1)
    # # KL_divergence = 0.5 * tf.reduce_sum(sigma - tf.square(mu) - tf.exp(sigma) , 1)
    # KL_divergence = tf.reduce_mean(KL_divergence)

    vars = tf.trainable_variables()
    loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name]) *3e-2

    # kl = tf.keras.losses.KLDivergence()
    # ki_loss = kl(outputLabel, y)
    # loss = tf.losses.mean_squared_error(outputLabel,y)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=outputLabel)
    # loss = tf.losses.categorical_crossentropy(y, outputLabel)
    loss = tf.multiply(loss, weights)
    loss = tf.reduce_mean(loss)
    loss = loss+loss_reg

    # ELBO = marginal_likelihood - KL_divergence



    correct_prediction = tf.equal(predictLabels, tf.argmax(outputLabel, axis=1))
    acc = tf.cast(correct_prediction, tf.float32)
    acc = tf.reduce_mean(acc)
    # train = tf.train.RMspropOptimizer(learning_rate=lr,rho=0.95, epsilon=1e-6).minimize(loss)
    train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    # train = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)


    trainOperaion = {'train': train, 'loss_total': loss, 'loss': loss, 'acc': acc,
                     'inputPC': inputPC,'loss_reg': loss_reg,
                     'inputGraph': inputGraph, 'outputLabel': outputLabel, 'weights': weights,
                     'predictLabels': predictLabels,
                     'keep_prob_1': keep_prob_1, 'keep_prob_2': keep_prob_2, 'lr': lr,
                     }

    return trainOperaion


def decoder(z, dim_img, n_hidden):
    y = bernoulli_MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)
    return y


def max_norm_regularizer(threshold, axes=1, name='max_nrom', collection='max-norm'):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weithts = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weithts)
        return None

    return max_norm