import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from pylab import *


def show_adj(adj,max,min):
    # f,ax1 = plt.subplots(figsize=(6, 4), nrows=1)
    #
    # cmap = sns.cubehelix_palette(start=0, rot=-.4, gamma=0.5, as_cmap=True)
    # sns.heatmap(adj, linewidths=0.05, ax=ax1, vmax=10, vmin=0, cmap=cmap)
    # ax1.set_title('adj')
    # ax1.set_xlabel('channels1')
    # ax1.set_ylabel('channels2')
    # plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(adj,cmap='GnBu')
    # fig.colorbar(cax)
    # ticks = np.arange(0,63, 10)
    # ax.set_xticks(ticks)
    # ax.set_yticks(ticks)
    # ax.set_xticklabels(ticks)
    # ax.set_yticklabels(ticks)
    # plt.show()


    # fig = plt.figure()
    # names = ['学科A', '学科B', '学科C', '学科D', '学业成败']
    # fig.figsize:(40,40) #图片大小为20*20
    # 以下代码用户显示中文
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(adj, cmap=plt.cm.RdYlBu_r, linewidths=0,  annot=False,)
    # cax = plt.gcf().axes[-1]
    # cax.tick_params(labelsize=20)
    # plt.xticks(np.arange(62) + 1, range(62))  # 横坐标标注点
    # plt.yticks(np.arange(62) + 1, range(62))  # 纵坐标标注点
    ticks = np.arange(0, 63, 10)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    plt.show()

def show_confusion_matrix(predict):
    # true = [[1,0,0],[0,1,0],[0,0,1]]
    # predict = [[0,0,1],[0,1,0],[0,0,1]]
    # true = np.array(true)
    # predict = np.array(predict)
    # true = true.reshape(-1,3)
    print('111')
    predict = predict.reshape(-1,3)
    total = predict.shape[0]
    matrix = np.zeros((3,3))
    # n1 = np.sum(true, axis=0)
    n2 = np.sum(predict, axis=0)

    # p = n2/n1
    for i in range(3):
        for j in range(3):
            matrix[i][j] = predict[i][j]/total
    print(matrix)
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(matrix, cmap=plt.cm.gray_r, linewidths=0.2, annot=True, )

    ticks = np.arange(0, 3, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    plt.show()



if __name__ == '__main__':
    a = np.random.randint(-4, 10, (62, 62))
    print(a)
    show_adj(a,10,0)
    # confusion_matrix()