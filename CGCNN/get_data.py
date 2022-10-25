import scipy.io as sio
import numpy as np
import pickle as pkl
from scipy import fftpack
from scipy.signal import butter, lfilter
from utils import scaled_laplacian,normalized_laplacian
from scipy.signal import detrend

sample_rate = 200
window_size = sample_rate*5

def dense_to_one_hot(labels, num_class):
    # labels = np.hstack(labels)
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_class
    label_one_hot = np.zeros((num_labels, num_class))
    label_one_hot.flat[index_offset + labels.ravel()] = 1
    return label_one_hot


def ex_feature(data_dict,label,key):
    # for k in range(5):
    trdata = []
    trlabels = []
    de_data = []
    for i in range(16):
        start = 0
        d = key+'{}'.format(i+1)
        a = data_dict[d]
        a = a.reshape(62,-1)
        # print('12333',a.shape)
        end = 10
        start = 0
        while end<a.shape[1]:
            tmp = a[:, start:end]
            de_data.append(tmp)
            trlabels.append(label[i])
            start = end
            end+=10
        # de_data = np.array(de_data)
        # print('de_data', de_data.shape)
        # trdata.append(de_data)
    de_data = np.array(de_data)
    trlabels = np.array(trlabels)
    print('trdata',de_data.shape,trlabels.shape)

    tedata = []
    telabels = []
    for i in range(16,24):
        start = 0
        d = key+'{}'.format(i+1)
        a = data_dict[d]
        a = a.reshape(62,-1)
        # print('12333',a.shape)
        end = 10
        start = 0
        while end<a.shape[1]:
            tmp = a[:, start:end]
            tedata.append(tmp)
            telabels.append(label[i])
            start = end
            end+=10
        # de_data = np.array(de_data)
        # print('de_data', de_data.shape)
        # trdata.append(de_data)
    tedata = np.array(tedata)
    telabels = np.array(telabels)
    print('tedata',tedata.shape,trlabels.shape)


    return trdata,trlabels,tedata,telabels


def load_data(subject):
    data_dict = sio.loadmat(r'/home/qq/data/' + subject + '.mat')
    label_dict = sio.loadmat(r'/home/qq/data/label.mat')
    label = label_dict['label']
    label += 1

    # seedIV
    # label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
    # label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
    # label = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]

    # n = int(subject[0])
    #
    # my_array = np.array([1, 1])
    # channel4 = [5, 6, 7, 12, 13, 14, 15, 23, 24, 25, 31, 32, 33, 34, 40, 41, 42, 43, 50]
    # data = []

    # for key in data_dict.keys():
    # de features
    # de_data,labels = ex_feature(data_dict,label,key='de_LDS')
    # labels = np.array(labels)
    # trdata, trlabels, tedata, telabels = split_data(de_data, labels,0.8)
    trdata,trlabels,tedata,telabels = ex_feature(data_dict,label,key='de_LDS')

    trlabels = np.array(trlabels)
    telabels = np.array(telabels)




    return trdata,trlabels,tedata,telabels

def plv_hilbert(ch1,ch2):
    hilbert1 = fftpack.hilbert(ch1)
    hilbert2 = fftpack.hilbert(ch2)
    l = len(ch1)
    phase1 = np.angle(hilbert1)
    phase2 = np.angle(hilbert2)
    unwrap1 = np.unwrap(phase1)
    unwrap2 = np.unwrap(phase2)
    values = 0
    for i in range(l):
        c = np.exp(complex(0, np.abs(unwrap1[i] - unwrap2[i])))
        values = values+c
        print('c', values)
    # plv = abs(sum(np.exp(c)))/l
    plv = abs(values)/l
    print('plv',plv)
    return plv

def newTimeCausality(ch1,ch2,order):
    #ch1、ch2:samples*1
    samples = len(ch1)
    ch1 = ch1.T
    ch2 = ch2.T
    M1 = np.zeros((samples-order,2*order))
    M2 = np.zeros((samples - order, 2 * order))
    for i in range(order,samples):
        M1[i-order,:] = np.concatenate((ch1[i-order:i],ch2[i-order:i]),axis=0)
        M2[i-order,:] = np.concatenate((ch2[i-order:i],ch1[i-order:i]),axis=0)
    y1 = ch1[order:].T
    y2 = ch2[order:].T

    #用最小二乘法仇联合回归系数
    # print('M1',M1,y1)
    # print(np.linalg.matrix_rank(M1))
    # c = np.dot(M1.T,M1)
    # t = np.mat(c)
    # print('c',c,c.shape,type(c))
    # a = np.linalg.inv(np.dot(M1.T,M1))
    # b = np.dot(M2.T,M2)
    # tt = np.mat(b)
    # print(t.I)
    coef_1 = np.dot(np.linalg.inv(np.dot(M1.T,M1)),np.dot(M1.T,y1))
    # print('coef',coef_1)
    coef_2 = np.dot(np.linalg.inv(M2.T*M2),np.dot(M2.T,y2))
    # print('coef',coef_1,M1)
    delt_1 = np.dot(M1,coef_1) - y1
    delt_2 = np.dot(M2,coef_2) - y2

    # avedelt_1 = np.mean(delt_1)
    # avedelt_2 = np.mean(delt_2)

    #计算第二个通道对第一个通道的影响值
    M1_2 = M1[:,order:]
    M1_1 = M1[:,:order]
    #第二个通道在第一个通道中占的分量
    ch2inch1part = np.dot(M1_2,coef_1[order:])
    #平方和
    ch2inch1partsquaresum = np.dot(ch2inch1part.T,ch2inch1part)
    ch1inch1part = np.dot(M1_1,coef_1[:order])
    ch1inch1partsquaresum = np.dot(ch1inch1part.T,ch1inch1part)

    #计算第一个通道对第二个通道的影响值
    M2_1 = M2[:,order:]
    M2_2 = M2[:,:order]

    ch1inch2part = np.dot(M2_1,coef_2[order:])
    ch1inch2partsquaresum = np.dot(ch1inch2part.T,ch1inch2part)
    ch2inch2part = np.dot(M2_2,coef_2[:order])
    ch2inch2partsquaresum = np.dot(ch2inch2part.T,ch2inch2part)

    ch1Toch2 = ch1inch2partsquaresum/(ch1inch2partsquaresum+ch2inch2partsquaresum+samples*np.var(delt_2))
    ch2Toch1 = ch2inch1partsquaresum/(ch1inch1partsquaresum+ch2inch1partsquaresum+samples*np.var(delt_1))
    # print('ch1Toch2',ch1Toch2,ch2Toch1)
    return ch1Toch2,ch2Toch1

def GrangerCausalityTime(ch1,ch2,order):
    #ch1、ch2:samples*1
    samples = len(ch1)
    ch1 = ch1.T
    ch2 = ch2.T
    M1 = np.zeros((samples-order,2*order))
    M2 = np.zeros((samples - order, 2 * order))
    for i in range(order,samples):
        M1[i-order,:] = np.concatenate((ch1[i-order:i],ch2[i-order:i]),axis=0)
        M2[i-order,:] = np.concatenate((ch2[i-order:i],ch1[i-order:i]),axis=0)
    y1 = ch1[order:].T
    y2 = ch2[order:].T

    #用最小二乘法仇联合回归系数
    coef_1 = np.dot(np.linalg.pinv(np.dot(M1.T,M1)),np.dot(M1.T,y1))
    # print('coef',coef_1)
    coef_2 = np.dot(np.linalg.pinv(np.dot(M2.T,M2)),np.dot(M2.T,y2))
    # print('coef',coef_1,M1)
    delt_1 = np.dot(M1,coef_1) - y1
    delt_2 = np.dot(M2,coef_2) - y2

    #自回归模型
    M1_a = np.zeros((samples - order, order))
    M2_a = np.zeros((samples - order, order))
    for i in range(order,samples):
        M1_a[i-order,:] = ch1[i-order:i]
        M2_a[i-order,:] = ch2[i-order:i]
    # print(M1_a.shape,y1.shape)
    coef_1a = np.dot(np.linalg.pinv(np.dot(M1_a.T,M1_a)),np.dot(M1_a.T,y1))
    coef_2a = np.dot(np.linalg.pinv(np.dot(M2_a.T,M2_a)),np.dot(M2_a.T,y2))
    # print('coef_1a',coef_1a,y1)
    # print('coef_2a',coef_2a.shape,y2.shape)
    delt_1a = np.dot(M1_a,coef_1a) - y1
    delt_2a = np.dot(M2_a,coef_2a) - y2
    # print('delt_1',np.var(delt_1),np.var(delt_1a))
    # print('delt_2',np.var(delt_2),np.var(delt_2a))
    ch1Toch2 = np.log(np.var(delt_2a)/np.var(delt_1))
    ch2Toch1 = np.log(np.var(delt_1a)/np.var(delt_2))

    return ch1Toch2,ch2Toch1

def preparedata(name):
    subject = name #'dujingcheng_20131107'
    # Data, Label = load_data(subject)
    train_x, train_y, test_x, test_y = load_data(subject)
    # print('1', Data.shape, Label.shape)
    # train_x, train_y, test_x, test_y = split_data(Data, Label,0.6)

    print('2', train_x.shape, train_y.shape)

    test_y = test_y.reshape(-1,1)
    train_y = train_y.reshape(-1,1)

    return  train_x, train_y, test_x, test_y

import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from plt import show_adj
def get_adj(data1,data2):
    data = np.concatenate((data1,data2), axis=0)
    # print('11111',data[0,0,:10])
    N = data.shape[1]
    M = data.shape[0]
    Data = []
    adj = dict()
    m1 = data1.shape[0]
    m2 = data2.shape[0]
    # m3 = data3.shape[0]
    adj1 = dict()
    adj2 = dict()
    adj3 = dict()
    Data = []

    # #identity matrix
    # graph = np.identity(N)

    # random matrix
    # graph = np.random.random((62, 62))

    #corr matrix
    # tempdata = data.reshape(N, -1)
    # graph = np.corrcoef(tempdata)

    # gelanjieyinguo matrix
    tempdata = data.reshape(N,-1)
    # print('tempdata',tempdata.shape)
    # tempdata = preprocessing.scale(tempdata)
    tempdata = detrend(tempdata)
    chmean = np.mean(tempdata)
    tempdata = tempdata - chmean
    tempdata = np.diff(tempdata)
    graph = np.zeros((N, N))
    for i in range(N):
        ch1 = tempdata[i, :]
        for j in range(i, N):
            ch2 = tempdata[j, :]
            # print('ch',ch1,ch2)
            ch1toch2, ch2toch1 = GrangerCausalityTime(ch1, ch2, 64)  # all order=3
            graph[i][j] = ch2toch1
            graph[j][i] = ch1toch2
    #
    # max = np.max(graph)
    # min = np.min(graph)
    # print(max, min)
    # # show_adj(graph, max, min)
    #
    for i in range(N):
        for j in range(i, N):
            graph[i][i] = 1
    # max = np.max(graph)
    # min = np.min(graph)
    # show_adj(graph, max,min)

    # save matrix
    # tmp = graph.reshape(-1,1)
    # import heapq
    # max = heapq.nlargest(10, tmp)
    # matrix = np.zeros((N, N))
    # for i in range(N):
    #     for j in range(N):
    #         if(graph[i][j] in max):
    #             matrix[i][j] = graph[i][j]
    #         else:
    #             matrix[i][j] = 0
    # from scipy.io import savemat

    # import xlwt
    # workbook = xlwt.Workbook()
    # sheet = workbook.add_sheet("Sheet")
    #
    # for i in range(N):
    #     for j in range(N):
    #         sheet.write(i, j, matrix[i][j])
    #
    # workbook.save("de_0.xls")
    # print('end save')
    graph = scaled_laplacian(graph)
    # graph = normalized_laplacian(graph)

    # print(graph.shape,type(graph), graph)
    # aaa = graph.A
    # # print(type(aaa))
    # show_adj(aaa,np.max(aaa),np.min(aaa))
    # graph = graph.reshape((1, N * N))
    # print('graph',graph[0])
    Data1 = []

    for i in range(m1):
        tempdata = data1[i, :, :]
        tempdata = preprocessing.scale(tempdata)
        # X_scaler = StandardScaler()
        # tempdata = X_scaler.fit_transform(tempdata)
        # Data1.append(tempdata)
        # Data1 = data1.reshape(62,-1)
        # X_scaler = StandardScaler()
        # X_train = X_scaler.fit_transform(tempdata)
        # tempdata = Z_ScoreNormalization(tempdata)
        # Data1 = Data1.reshape(m1,62,-1)
        Data1.append(tempdata)
        adj1.update({i:graph})

    Data2 = []

    for i in range(m2):
        tempdata = data2[i,:,:]
        tempdata = preprocessing.scale(tempdata)
        # scaler.data_max_
        # 标准化数据
        # tempdata = scaler.transform(tempdata)
        # X_scaler = StandardScaler()
        # tempdata = X_scaler.fit_transform(tempdata)
        Data2.append(tempdata)
        # Data2 = data2.reshape(62, -1)
        # X_scaler = StandardScaler()
        # X_train = X_scaler.fit_transform(tempdata)
        # tempdata = Z_ScoreNormalization(tempdata)
        # Data2.append(tempdata)
        # Data2 = Data2.reshape(m2,62,-1)
        adj2.update({i:graph})


    Data1 = np.array(Data1)
    Data2 = np.array(Data2)
    # Data3 = np.array(Data3)
    # print('111',Data2[2,1,:])
    m = []
    # for k in range(M):
    #     tempdata = data[k,:,:]
    #     # a = find_martrix_max_value(tempdata)
    #     # m.append(a)
    #     # tempdata = preprocessing.scale(tempdata)
    #     # X_scaler = StandardScaler()
    #     # tempdata = X_scaler.fit_transform(tempdata)
    #     # tempdata = Z_ScoreNormalization(tempdata)
    #     # tempdata = tempdata/38
    #     Data.append(tempdata)
    #     tempdata = detrend(tempdata)
    #     # # print('tempdata',tempdata.shape)
    #     chmean = np.mean(tempdata)
    #     tempdata = tempdata - chmean
    #     tempdata = np.diff(tempdata)
    #     # print('r_tempdata',np.linalg.matrix_rank(tempdata))
    #     # a = abs(np.corrcoef(tempdata))
    #     # graph.append(a)
    #     graph = np.zeros((N,N))
    #     for i in range(N):
    #         ch1 = tempdata[i, :]
    #         for j in range(i, N):
    #             ch2 = tempdata[j, :]
    #             # print('ch',ch1,ch2)
    #             ch1toch2, ch2toch1 = GrangerCausalityTime(ch1,ch2,2)#all order=3
    #             graph[i][j] = ch2toch1
    #             graph[j][i] = ch1toch2
    #             # plv = plv_hilbert(ch1, ch2)
    #             # graph[i][j] = plv
    #             # graph[j][i] = plv
    #     for i in range(N):
    #         for j in range(i, N):
    #             graph[i][i] = 1
    #     # print('graph',graph[6])
    #     graph = scaled_laplacian(graph)
    #     # print('graph',graph[15])
    #     graph = graph.reshape((1,N*N))
    #
    #     adj.update({k: graph})
    #     # adj[k] = graph
    # # adj = np.array(adj)
    # # print('abc', adj[1],adj[2])
    # Data = np.array(Data)
    # print('max',np.max(Data),np.min(Data))
    return Data1,adj1,Data2,adj2

