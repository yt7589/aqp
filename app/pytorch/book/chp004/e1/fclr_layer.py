# 全连接层并用采用LeakyReLU作为激活函数
import os
import math
import copy
import pickle
import numpy as np

class FclrLayer(object):
    def __init__(self, optimizer, K, N, epsilon=0.2, param_file='work/fclr.pkl'):
        '''
        参数：
          K：本层神经元个数
          N: 特征维度（下一层神经元数）
        '''
        self.epsilon = epsilon
        self.name = 'ann.layer.FclrLayer'
        self.X = None
        self.Y = None
        self.Y_ = None
        self.W = None # 连接权值
        self.b = None # 偏置值
        self.W_opt  = None
        self.b_opt = None
        self.K = K
        self.N = N
        self.input_shape = (N,)
        self.trainable = True # 是否参加训练过程
        self.activation = self.leaky_relu
        if os.path.exists(param_file):
            self.can_restore_layer = True
        else:
            self.can_restore_layer = False
        self.param_file = param_file

    def layer_name(self):
        return '全连接Leaky_ReLU'

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.b.shape)

    def output_shape(self):
        return (self.K, )

    def save_layer(self):
        params = [self.W, self.b]
        with open(self.param_file, 'wb') as fd:
            pickle.dump(params, fd)

    def restore_layer(self):
        with open(self.param_file, 'rb') as fd:
            params = pickle.load(fd)
        self.W = np.array(params[0])
        self.b = np.array(params[1])

    def initialize(self, optimizer):
        '''
        初始化网络参数
        '''
        if self.can_restore_layer:
            self.restore_layer()
        else:
            # Initialize the weights
            limit = 1 / math.sqrt(self.N)
            self.W  = np.random.uniform(-limit, limit, (self.K, self.N))
            self.b = np.zeros((self.K, 1))
            # Weight optimizers
            self.W_opt  = copy.copy(optimizer)
            self.b_opt = copy.copy(optimizer)

    def leaky_relu(self, X):
        return np.where(X >= 0, X, self.epsilon * X)

    def forward_pass(self, X, Y, training=True):
        '''
        前向传播过程
        参数：
          X：输入信号，M*N，其中M为迷你批次大小
          Y：正确值，one-hot向量形式，M*K
        '''
        Z = X.dot(np.transpose(self.W)) + np.transpose(self.b)
        Y_ = self.activation(Z)
        self.X = X
        self.Y = Y
        self.Y_ = Y_
        return Z, Y_

    def backward_pass(self, accum_grad):
        org_W = self.W
        M, _ = self.X.shape
        # 求出leaky_relu的微分
        self.Y_[self.Y_>0] = 1
        self.Y_[self.Y_<=0] = self.epsilon
        pJ_pW_raw = None
        pJ_pb_raw = None
        pJ_pX = None
        for i in range(M):
            gi = accum_grad[i, :]
            ai = self.Y_[i, :]
            gai = gi * ai
            gvw = gai.dot(org_W)
            if self.trainable:
                gai = gai.reshape((self.K, 1))
                xi = self.X[i, :].reshape((1, self.N))
                gai_xi = gai.dot(xi)
                if pJ_pW_raw is None:
                    pJ_pW_raw = np.array([gai_xi])
                else:
                    pJ_pW_raw = np.append(pJ_pW_raw, [gai_xi], axis=0)
                if pJ_pb_raw is None:
                    pJ_pb_raw = np.array([gai])
                else:
                    pJ_pb_raw = np.append(pJ_pb_raw, [gai], axis=0)
            if pJ_pX is None:
                pJ_pX = np.array([gvw])
            else:
                pJ_pX = np.append(pJ_pX, [gvw], axis=0)
        if self.trainable:
            pJ_pW = np.sum(pJ_pW_raw, axis=0)
            pJ_pb = np.sum(pJ_pb_raw, axis=0)
            self.W = self.W_opt.update(self.W, pJ_pW)
            self.b = self.b_opt.update(self.b, pJ_pb)
        return pJ_pX