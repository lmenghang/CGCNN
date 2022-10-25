
class Parameters():
    def __init__(self):
        self.inputFeatureN = 20
        # self.neighborNumber = 30
        self.outputClassN = 4
        self.pointNumber = 62#23#27#de62
        self.intermediate_dim = 64
        self.latent_dim = 16
        self.gcn_1_filter_n = 64 #de128
        self.gcn_2_filter_n = 64#de_all128
        self.gcn_3_filter_n = 256
        self.fc_1_n = 128
        self.chebyshev_1_Order = 3
        self.chebyshev_2_Order = 5
        self.keep_prob_1 = 0.5 #de_all0.9
        self.keep_prob_2 = 0.5#de_all 0.8
        self.batchSize = 20#de_all 30
        self.testBatchSize = 20
        self.max_epoch = 50
        self.learningRate = 0.01
        self.dataset = 'SEED-IV'
        self.weighting_scheme = 'weighted'
        self.modelDir = '/model/'
        self.logDir = '/log/'
        self.fileName = '2gcn_cheby_3_2'
        self.weight_scaler = 4#de_all30
        # self.N1 = 10  # # of nodes belong to each window
        # self.N2 = 10  # # of windows -------Feature mapping layer 窗口数
        # self.N3 = 500  # # of enhancement nodes -----Enhance layer
        # self.L = 5  # # of incremental steps 增量步骤
        # self.M1 = 50  # # of adding enhance nodes
        # self.s = 0.8  # shrink coefficient 收缩系数
        # self.C = 2 ** -30  # Regularization coefficient 正则化系数


