#
import numpy as np
import torch

class Chp001C007(object):
    def __init__(self):
        self.name = 'app.pytorch.book.chp001.Chp001C007'

    def run(self):
        # cifar-10 16个样本为一个批次
        X = torch.rand(16, 3, 32, 32)
        # from zero start_idx(inclusive); end_idx(exclusive); step步长
        # 当为负数时由-1开始，表示是最后一个元素
        print('前两张图：{0}'.format(X[:2, :, :, :].shape))
        print('最后一张图：{0}'.format(X[-1:, :, :, :].shape))
        print('取偶数图片：{0}'.format(X[1::2, :, :, :].shape))
        print('选第1、8、9张图：{0}'.format(X.index_select(0, 
                    torch.tensor([0, 7, 8])).shape))
        print('取R和B两通道：{0}'.format(X.index_select(1, 
                    torch.tensor([0, 2])).shape))
        print('取每张图片8*8区域：{0}'.format( X.index_select(2, 
                    torch.arange(8)).index_select(3, 
                    torch.arange(6)).shape ))
        print('语法糖：第1张图：{0}; 颜色G通道：{1}'.format(
            X[0, ...].shape, X[:, 1, ...].shape)
        )
        mask = X.ge(0.5)
        print('大于0.5的元素：{0}'.format(
            torch.masked_select(X, mask))
        )
        t2 = torch.tensor([[3.0, 3.1, 3.2], [3.3, 3.4, 3.5]])
        print('按1维索引取值：{0}'.format(torch.take(t2, 
                    torch.tensor([0, 1, 5]))))