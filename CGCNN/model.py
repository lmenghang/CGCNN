# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
from get_data import dense_to_one_hot
from layers import gcnLayer, globalPooling, fullyConnected
from utils import get_mini_batch, add_noise, weights_calculation, uniform_weight


# ===========================Hyper parameter=====================



# (input data) input inputCoor, input Graph, input Label dictionary -------3
# (model training)train, loss, acc,loss_l2 -------------3
# (hyper parameter)batchSize, keep_prob, keep_prob_1,lr:learning_rate
# Return average loss, acc, reg


from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize


def trainOneEpoch(inputCoor, inputGraph, inputLabel, para, sess, trainOperaion, weight_dict, learningRate):
    dataChunkLoss = []
    dataChunkAcc = []
    dataChunkRegLoss = []
    trainpredict = []
    # inputGraph = inputGraph.tocsr()
    # print('inputGraph',inputGraph.shape)
    for i in range(1):
        # start = i*para.batchSize
        # end = start+para.batchSize
        # xTrain_1, graphTrain_1, labelTrain_1 = get_mini_batch(inputCoor,inputGraph,inputLabel,start,end)#inputCoor[i], inputGraph[i], inputLabel[i]
        # graphTrain_1 = graphTrain_1.tocsr()
        labelBinarize = label_binarize(inputLabel, classes=[j for j in range(para.outputClassN)]) #one-hot编码
        # print('shape',xTrain_1.shape,graphTrain_1.shape,labelBinarize.shape)
        xTrain, graphTrain, labelTrain = shuffle(inputCoor, inputGraph, labelBinarize)
        # labelBinarize = label_binarize(labelTrain, classes=[j for j in range(40)])

        batch_loss = []
        batch_acc = []
        batch_reg = []
        batchSize = para.batchSize
        for batchID in range(len(labelTrain) // para.batchSize):
            start = batchID * batchSize
            end = start + batchSize
            batchCoor, batchGraph, batchLabel = get_mini_batch(xTrain, graphTrain, labelTrain, start, end)
            # print('batchGraph',batchGraph.shape)
        #     batchGraph = batchGraph.todense()
        #     batchGraph = graphTrain
        #     batchCoor = xTrain
        #     batchLabel = labelTrain
        # print('batchGraph',batchGraph)
        # batchGraph = tf.reshape(batchGraph, [-1, para.pointNumber, para.pointNumber])
        # D = []
        # k = 0
        # for j in range(batchGraph.shape[0]):
        #     temp = batchGraph[j]
        #     D.append(np.sum(temp[k]))
        #     k = k+1
        # value = tf.reduce_sum(batchGraph,reduction_indices=2)
        # for j in range(value.shape[0]):
        #     D.append(tf.diag(value[j]))
        # D_ = np.power(D, -1 / 2)
        # D = tf.convert_to_tensor(D)
        # batchGraph = D-batchGraph
        # # D = tf.convert_to_tensor(D,tf.float32)
        # support = tf.matmul(D_,batchGraph)
        # support = tf.matmul(support,D_)

            batchCoor = add_noise(batchCoor, sigma=0.08, clip=0.2)
            if para.weighting_scheme == 'uniform':
                batchWeight = uniform_weight(batchLabel)
            elif para.weighting_scheme == 'weighted':
                batchWeight = weights_calculation(batchLabel, weight_dict)
            else:
                print('please enter the valid weighting scheme')


            feed_dict = {trainOperaion['inputPC']: batchCoor, trainOperaion['inputGraph']: batchGraph,
                         trainOperaion['outputLabel']: batchLabel, trainOperaion['lr']: learningRate,
                         trainOperaion['weights']: batchWeight,
                         trainOperaion['keep_prob_1']: para.keep_prob_1, trainOperaion['keep_prob_2']: para.keep_prob_2}

            opt, loss_train, acc_train,loss_reg_train, predict = sess.run(
                [trainOperaion['train'], trainOperaion['loss_total'], trainOperaion['acc'],trainOperaion['loss_reg'],trainOperaion['predictLabels']],
                feed_dict=feed_dict)



        #print('The loss loss_reg and acc for this batch is {},{} and {}'.format(loss_train, loss_reg_train, acc_train))
            batch_loss.append(loss_train)
            batch_acc.append(acc_train)
            batch_reg.append(loss_reg_train)
            trainpredict.append(predict)

        dataChunkLoss.append(np.mean(batch_loss))
        dataChunkAcc.append(np.mean(batch_acc))
        dataChunkRegLoss.append(np.mean(batch_reg))
        # dataChunkAcc.append(acc_train)
        # dataChunkLoss.append(loss_train)
        # dataChunkRegLoss.append(loss_reg_train)

    train_average_loss = np.mean(dataChunkLoss)
    train_average_acc = np.mean(dataChunkAcc)
    loss_reg_average = np.mean(dataChunkRegLoss)
    return train_average_loss, train_average_acc,loss_reg_average, trainpredict


def evaluateOneEpoch(inputCoor, inputGraph, inputLabel, para, sess, trainOperaion):
    test_loss = []
    test_acc = []
    test_predict = []

    labelBinarize = label_binarize(inputLabel, classes=[j for j in range(para.outputClassN)])
    xTest_, graphTest_, labelTest_ = shuffle(inputCoor, inputGraph, labelBinarize)
    batchCoor, batchGraph, batchLabel = get_mini_batch(xTest_, graphTest_, labelTest_, 0, len(labelTest_))
    batchWeight = uniform_weight(batchLabel)
    feed_dict = {trainOperaion['inputPC']: batchCoor, trainOperaion['inputGraph']: batchGraph,
                 trainOperaion['outputLabel']: batchLabel, trainOperaion['weights']: batchWeight,
                 trainOperaion['keep_prob_1']: 1.0, trainOperaion['keep_prob_2']: 1.0}

    predict, trueLabel, loss_test, acc_test = sess.run(
        [trainOperaion['predictLabels'], trainOperaion['outputLabel'], trainOperaion['loss'], trainOperaion['acc']],
        feed_dict=feed_dict)
    # for i in range(1):
    #     # start = i * para.testBatchSize
    #     # end = start + para.testBatchSize
    #     # xTest, graphTest, labelTest = inputCoor[i], inputGraph[i], inputLabel[i]
    #     # xTest, graphTest, labelTest = get_mini_batch(inputCoor, inputGraph, inputLabel, start, end)
    #     # graphTest = graphTest.tocsr()
    #     labelBinarize = label_binarize(inputLabel, classes=[j for j in range(para.outputClassN)])
    #     xTest_, graphTest_, labelTest_ = shuffle(inputCoor, inputGraph, labelBinarize)
    #     test_batch_size = para.testBatchSize
    #     for testBatchID in range(len(labelTest_) // para.testBatchSize):
    #         start = testBatchID * test_batch_size
    #         end = start + test_batch_size
    #         batchCoor, batchGraph, batchLabel = get_mini_batch(xTest_, graphTest_, labelTest_, start, end)
    #         batchWeight = uniform_weight(batchLabel)
    #     # batchGraph = graphTest_
    #     # batchCoor = xTest_
    #     # batchLabel = labelTest_
    #     # batchCoor = add_noise(batchCoor, sigma=0.008, clip=0.02)
    #     #     batchCoor = add_noise(batchCoor, sigma=0.08, clip=0.2)
    #         feed_dict = {trainOperaion['inputPC']: batchCoor, trainOperaion['inputGraph']: batchGraph,
    #                      trainOperaion['outputLabel']: batchLabel, trainOperaion['weights']: batchWeight,
    #                      trainOperaion['keep_prob_1']: 1.0, trainOperaion['keep_prob_2']: 1.0}
    #
    #         predict, trueLabel, loss_test, acc_test = sess.run(
    #             [trainOperaion['predictLabels'], trainOperaion['outputLabel'], trainOperaion['loss'], trainOperaion['acc']], feed_dict=feed_dict)
    #         test_loss.append(loss_test)
    #         test_acc.append(acc_test)
    #         test_predict.append(predict)
    #         trueLabel = tf.argmax(trueLabel, axis=1)
    #     # print(predict.shape, trueLabel.shape)

    # test_average_loss = np.mean(test_loss)
    # test_average_acc = np.mean(test_acc)


    # return test_average_loss, test_average_acc, predict,trueLabel
    return loss_test, acc_test, predict,trueLabel
