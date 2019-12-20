#
import numpy as np
import torch

class Chp001C006(object):
    def __init__(self):
        self.name = 'app.pytorch.book.chp001.Chp001C006'

    def run(self):
        t1 = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        print('全部和：{0}; 行和：{1}; 列合：{2}'.format(t1.sum(), 
                    t1.sum(dim=1), t1.sum(dim=0)))
        print('均值：{0}; 行：{1}; 列：{2}'.format(t1.mean(), 
                    t1.mean(dim=1), t1.mean(dim=0)))
        print('方差：{0}; 行：{1}; 列：{2}'.format(t1.std(), 
                    t1.std(dim=1), t1.std(dim=0)))
        print('最大值：全部：{0}; 行：{1}; 列：{2}'.format(t1.max(), 
                    torch.max(t1, 1), torch.max(t1, 0)))
        t2 = torch.tensor([
            [0.1, 0.5, 0.4],
            [0.9, 0.05, 0.05]
        ])
        print('类别：{0}'.format(t2.argmax(dim=1)))