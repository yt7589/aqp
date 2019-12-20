#
import numpy as np
import torch 

class Chp001C004(object):
    def __init__(self):
        self.name = 'app.pytorch.book.chp001.Chp001C004'

    def run(self):
        t1 = torch.tensor([
            [1.1, 1.2, 1.3],
            [2.1, 2.2, 2.3],
            [3.1, 3.2, 3.3],
            [4.1, 4.2, 4.3]
        ])
        print('t1.shape:{0}'.format(t1.shape))
        t2 = t1.reshape(2, 6)
        print('t2.shape:{0}\r\n{1}'.format(t2.shape, t2))
        t3 = t1.reshape(-1, 4)
        print('t3.shape:{0}\r\n{1}'.format(t3.shape, t3))
        print('元素个数：t1={0}; t2={1}'.format(t1.numel(), t2.numel()))
        t4 = t1.reshape(1, -1).squeeze()
        print('t4:{0}; {1}'.format(t4.shape, t4))
        t7 = torch.rand(28, 28)
        t8 = t7.flatten()
        print(t8.shape)
        t5 = t1.unsqueeze(1)
        print('t5:{0}; {1}'.format(t5.shape, t5))
        t6 = t5.squeeze(1)
        print('t6:{0}; {1}'.format(t6.shape, t6))
        img1 = torch.rand(28, 28)
        img2 = torch.rand(28, 28)
        img3 = torch.rand(28, 28)
        X_raw = torch.cat((img1, img2, img3), dim=0).reshape(3, 28, 28)
        print('X_raw:{0}'.format(X_raw.shape))
        X = X_raw.unsqueeze(1)
        print('X:{0}'.format(X.shape))
        a0 = X.flatten(start_dim=1)
        print('a0.shape:{0}'.format(a0.shape))

    def my_flatten(t):
        return t.reshape(1, -1).squeeze()