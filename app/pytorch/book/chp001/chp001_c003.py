#
import numpy as np
import torch

class Chp001C003(object):
    def __init__(self):
        self.name = 'app.pytorch.book.chp001.Chp001C003'

    def run(self):
        v1 = np.array([
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
            [31.0, 32.0, 33.0, 34.0, 35.0]
        ], dtype=np.float32)
        t1 = torch.from_numpy(v1)
        print('t1:{0}'.format(t1))
        t2 = torch.tensor([101.0, 202.0, 303.0])
        v2 = t2.numpy()
        print('v2:{0}'.format(v2))
        v1[0][0] = 0.0
        v1[0][1] = 0.0
        print('修改后的t1：{0}'.format(t1))
        t2[0] = 0.0
        print('修改后的v2：{0}'.format(v2))