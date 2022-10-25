# from read_data import load_data, prepareData
# import tensorflow as tf
import tensorflow.compat.v1 as tf
# from sklearn.externals import joblib

tf.disable_eager_execution()
from model import trainOneEpoch, evaluateOneEpoch
# from gcn_lpa import model_architecture
# from gcn import model_architecture

import numpy as np
from Parameters import Parameters
from utils import weight_dict_fc,scaled_laplacian
from sklearn.metrics import confusion_matrix
from plt import show_confusion_matrix
import time
from get_data import preparedata,get_adj
# from GC import get_adj
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
# config.gpu_options.allow_growth = True

start_time = time.time()
samplingType = 'farthest_sampling'
# ===============================Hyper parameters========================
para = Parameters()
print('Dataset {}'.format(para.dataset))
print('The point number is {} '.format(para.pointNumber))
print('The first and second layer filter number is {} and {}'.format(para.gcn_1_filter_n, para.gcn_2_filter_n))
print('The fc neuron number is {} and the output number is {}'.format(para.fc_1_n, para.outputClassN))
print('The Chebyshev polynomial order for each layer are {} and {}'.format(para.chebyshev_1_Order, para.chebyshev_2_Order) )
print('The weighting scheme is {} and the weighting scaler is {}'.format(para.weighting_scheme, para.weight_scaler))


#=======================================================================

pointNumber = para.pointNumber
# neighborNumber = para.neighborNumber
results = []
from sklearn.preprocessing import label_binarize
from sklearn import svm
with tf.Graph().as_default():
    # names=['dujingcheng_20131027','dujingcheng_20131030']
    # names = ['dujingcheng_20131107','jingjing_20140603','liuqiujun_20140702',
    # names = ['jingjing_20140611','jingjing_20140629']
    # names=['liuqiujun_20140621','liuqiujun_20140705','liuye_20140411','liuye_20140418',
    names = ['liuqiujun_20140705',]
           # 'liuye_20140506','mahaiwei_20130712','mahaiwei_20131016','mahaiwei_20131113',]
    # names=['penghuiling_20131027','penghuiling_20131030','penghuiling_20131106','sunxiangyu_20140511'
    # ,'sunxiangyu_20140514','sunxiangyu_20140521','wangkui_20140620','wangkui_20140627','wangkui_20140704','weiwei_20131130','weiwei_20131204']
    # names=['weiwei_20131211','wusifan_20140618','wusifan_20140625','wusifan_20140630'
    # ,'wuyangwei_20131127','wuyangwei_20131201','wuyangwei_20131207',]
    # names=['xiayulu_20140527','xiayulu_20140603','xiayulu_20140610','yansheng_20140601','yansheng_20140615','yansheng_20140627','zhujiayi_20130709','zhujiayi_20131016','zhujiayi_20131105']

    # seed-IV
    # names = ['1_20160518','2_20150915','3_20150919','4_20151111','5_20160406','6_20150507','7_20150715','8_20151103','9_20151028','10_20151014','11_20150916','12_20150725','13_20151115','14_20151205','15_20150508']
    # names = ['1_20161125','2_20150920','3_20151018','4_20151118','5_20160413','6_20150511','7_20150717','8_20151110','9_20151119','10_20151021','11_20150921','12_20150804','13_20151125','14_20151208','15_20150514']
    # names = ['1_20161126','2_20151012','3_20151101','4_20151123','5_20160420','6_20150512','7_20150721','8_20151117','9_20151209','10_20151023','11_20151011','12_20150807','13_20161130','14_20151215','15_20150527']
    # names = ['5_20160406']
    predict = []
    y_true = []
    for id in names:
        print('subject:',id)
        test_acc_record = []
        test_mean_acc_record = []
        for i in range(5):
            print('subject:', id)
            ave_acc = []
        # ===============================Build model=============================
        #     trainOperaion = model_architecture(para)
            trainOperaion = autoencoder(para)
            #init = tf.global_variables_initializer()
            #sess = tf.Session()
            #sess.run(init)
            # ================================Load data===============================
            # if para.dataset == 'ModelNet40':
            #     inputTrain, trainLabel, inputTest, testLabel = load_data(pointNumber, samplingType)
            # elif para.dataset == 'ModelNet10':
            #     ModelNet10_dir = '/raid60/yingxue.zhang2/ICASSP_code/data/'
            #     with open(ModelNet10_dir+'input_data','rb') as handle:
            #         a = pickle.load(handle)
            #     inputTrain, trainLabel, inputTest, testLabel = a
            # else:
            #     print("Please enter a valid dataset")

            inputTrain, trainLabel, inputTest, testLabel = preparedata(id)
            # inputTest,testLabel,inputVal,valLabel = split_data(inputTest,testLabel,0.5)
            # inputTrain,adjtrain = get_adj(inputTrain)
            # inputTest,adjtest = get_adj(inputTest)
            # inputVal,adjval = get_adj(inputVal)
            inputTrain, adjtrain,inputTest,adjtest = get_adj(inputTrain, inputTest)
            # scaledLaplacianVal = adjval
            scaledLaplacianTrain = adjtrain
            scaledLaplacianTest = adjtest
            # scaledLaplacianTrain, scaledLaplacianTest = prepareData(inputTrain, inputTest, neighborNumber, pointNumber)

            # ===============================Train model ================================
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
            saver = tf.train.Saver()
            learningRate = para.learningRate

            modelDir = para.modelDir
            save_model_path = modelDir + "model_" + para.fileName
            weight_dict = weight_dict_fc(trainLabel, para)
            # valLabelWhole = []
            # for i in range(len(valLabel)):
            #     labels = valLabel[i]
            #     [valLabelWhole.append(j) for j in labels]
            # valLabelWhole = np.asarray(valLabelWhole)


            for epoch in range(para.max_epoch):
                print('===========================epoch {}===================='.format(epoch+1))
                # if (epoch % 10 == 0):
                #     learningRate = learningRate /2 #1.7
                # learningRate = np.max([learningRate, 0.0001]) #de_all0.000125
                print(learningRate)
                train_average_loss, train_average_acc,loss_reg_average, train_predict= trainOneEpoch(inputTrain, scaledLaplacianTrain,
                                                                                        trainLabel,
                                                                                        para, sess, trainOperaion, weight_dict,
                                                                                        learningRate)

                # save = saver.save(sess, save_model_path)
                # print('average loss:{}, l2 loss;{}, acc: {}  for this epoch======'.format(train_average_loss,
                #                                                                          loss_reg_average,
                #                                                                          train_average_acc))
                # val_average_loss, val_average_acc, val_predict = evaluateOneEpoch(inputVal, scaledLaplacianVal, valLabel,
                #                                                                      para, sess, trainOperaion)
                test_average_loss, test_average_acc, test_predict, trueLabel = evaluateOneEpoch(inputTest, scaledLaplacianTest,
                                                                                     testLabel,
                                                                                     para, sess, trainOperaion)
                # print(test_predict)
                # print(trueLabel)

                # predict.append(test_predict)
                # y_true.append(trueLabel)
                # calculate mean class accuracy
                # val_predict = np.asarray(val_predict)
                # val_predict = val_predict.flatten()
                # confusion_mat = confusion_matrix(valLabelWhole[0:len(val_predict)], val_predict)
                # normalized_confusion = confusion_mat.astype('float') / confusion_mat.sum(axis=1)
                # class_acc = np.diag(normalized_confusion)
                # mean_class_acc = np.mean(class_acc)

                # save log
                log_Dir = para.logDir
                fileName = para.fileName
                # with open(log_Dir + 'confusion_mat_' + fileName, 'wb') as handle:
                #     pickle.dump(confusion_mat, handle)
                # print('the average acc among 3 class is:{}'.format(mean_class_acc))
                # train_predict = np.array(train_predict)
                # train_predict = train_predict.reshape(-1, 3)
                # test_predict = np.array(test_predict)
                # test_predict = test_predict.reshape(-1, 3)
                # print('yyyy', test_predict.shape, testLabel.shape)
                # clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
                # clf.fit(train_predict, trainLabel)
                # train_acc = clf.score(train_predict,trainLabel)
                # test_acc = clf.score(test_predict,testLabel)

                print('average loss:{}, l2 loss;{}, acc: {}  for this epoch======'.format(train_average_loss,
                                                                                          loss_reg_average,
                                                                                          train_average_acc))
                print('========test average loss: {} and acc: {} for this model ======='.format(test_average_loss,
                                                                                                test_average_acc))
                # print(trueLabel)
                # trueLabel = tf.convert_to_tensor(trueLabel)
                # print('predict:sad', np.sum(np.array(test_predict)==0),'neural',np.sum(np.array(test_predict)==1),'happy',np.sum(np.array(test_predict)==2))
                # print('true:sad', np.sum(trueLabel.eval(session=sess)==0),'neural',np.sum(trueLabel.eval(session=sess)==1),'happy',np.sum(trueLabel.eval(session=sess)==2))
                # print(
                # 'average loss: {} and acc: {} for this epoch ======='.format(val_average_loss, val_average_acc))
                # test_acc_record.append(val_average_acc)
                # test_mean_acc_record.append(mean_class_acc)

                # with open(log_Dir + 'overall_acc_record_' + fileName, 'wb') as handle:
                #     pickle.dump(test_acc_record, handle)
                # with open(log_Dir + 'mean_class_acc_record_' + fileName, 'wb') as handle:
                #     pickle.dump(test_mean_acc_record, handle)
            # joblib.dump(trainOperaion, "gcn_model.m")
            # model = joblib.load("gcn_model.m")
            # test_average_loss, test_average_acc, test_predict = evaluateOneEpoch(inputTest, scaledLaplacianTest, testLabel,
            #                                                                          para, sess, trainOperaion)
            # print('========test average loss: {} and acc: {} for this model ======='.format(test_average_loss, test_average_acc))
            #     if epoch > 5:
                ave_acc.append(test_average_acc)

            test_acc_record.append(np.mean(ave_acc))
            print('{}: acc:{}'.format(i, np.mean(ave_acc)))
        # predict = np.array(predict)
        y_true = np.array(y_true)
        # predict = predict.reshape(-1,3)
        # y_true = y_true.reshape(-1,3)
        # print(y_true.shape, predict.shape)
        # predict = predict.reshape(-1,3)
        # confusion_mat = confusion_matrix(y_true, predict)
        # print(confusion_mat.shape, confusion_mat)

        mean = np.mean(test_acc_record)
        # if(mean<60):
        #     mean = np.max(test_acc_record)
        print('{}: mean acc:{}'.format(id, mean))
        results.append(mean)
# np.savetxt("log/seed/gcn2.csv", np.array(test_acc_record),fmt='%.4f', delimiter=',')
results = np.array(results)
print(results)
# print(np.mean(results))
# pd.DataFrame(results,index=[0]).to_csv('log/seed/gcn2.csv')
end_time = time.time()
run_time = (end_time - start_time)/3600
print('The running time for this trail is {} hours'.format(run_time))