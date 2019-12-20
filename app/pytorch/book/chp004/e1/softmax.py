#
import numpy as np

class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)
        
'''
class Softmax(object):
    def __call__(self, z):
        z_max = z.max(axis=1) # 求出每个样本K维中的最大值
        z_ = z_max.reshape(z_max.shape[0],1) # 变为(m,1)维
        z -= z_ # 将m个样本的K维输入值减去该样本K维的最大值
        ez = np.exp(z) # 求出指数值
        # 求出每个样本K维指数之和
        dsum = np.sum(ez, axis=1).reshape(z_max.shape[0], 1)
        return ez / dsum # 返回softmax函数值

    def gradient(self, y, z):
        m = y.shape[0]
        K = y.shape[1]
        grad = np.zeros((m, K, K))
        for i in range(m):
            for j in range(K):
                for k in range(K):
                    if j is not k:
                        grad[i, j, k] = -y[i, k]*y[i, j]
                    else:
                        grad[i, j, k] = y[i, k]*(1-y[i, k])
        return grad
'''