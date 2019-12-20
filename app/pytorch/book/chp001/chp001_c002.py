#
import numpy as np
import torch

class Chp001C002(object):
    def __init__(self):
        self.name = 'app.pytorch.book.chp001.Chp001C002'

    def run(self):
        t1 = torch.tensor([
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
            [31.0, 32.0, 33.0, 34.0, 35.0]
        ])
        print('t1 size:{0}; shape:{1}; axis0:{2}'.format(t1.size(), t1.shape, t1.shape[0]))
        t2 = torch.tensor([
            [41.0, 42.0, 43.0, 44.0, 45.0]
        ])
        t3 = torch.cat((t1, t2), dim=0)
        print('t3 size:{0}\n{1}'.format(t3.shape, t3))
        t4 = torch.tensor([
            [101.0],
            [102.0],
            [103.0],
            [104.0]
        ])
        t5 = torch.cat((t3, t4), dim=1)
        print('t5 size:{0}\n{1}'.format(t5.shape, t5))