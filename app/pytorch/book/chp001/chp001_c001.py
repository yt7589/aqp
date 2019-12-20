#
import numpy as np
import torch

class Chp001C001(object):
    def __init__(self):
        self.name = 'app.pytorch.book.chp001.Chp001C001'

    def run(self):
        self.test()

    def create(self):
        t1 = torch.empty(5)
        t2 = torch.empty([5, 3]) #创建5*3空张量，元素值为随机数
        t3 = torch.rand(2, 3) #采用0~1的均匀分布随机数初始化二维张量
        t4 = torch.randn(2, 3) # 以均值为0，方差为1的正态分布初始化二维张量
        t5 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]) # 最常用方法
        t6 = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0]) # 利用构造函数
        # 元素全为1的张量
        t7 = torch.ones(2, 3, dtype=torch.float32) 
        t7_1 = torch.ones_like(t2) # 生成5*3二维张量，元素全为1
        # 创建元素值全为0的张量
        t8 = torch.zeros(2, 3, dtype=torch.int)
        t8_1 = torch.zeros_like(t2)

        print('t1:{0}'.format(t1))
        print('t2:{0}'.format(t2))
        print('t3:{0}'.format(t3))
        print('t4:{0}'.format(t4))
        print('t5:{0}'.format(t5))
        print('t6:{0}'.format(t6))
        print('t7:{0}'.format(t7))
        print('t7_1:{0}'.format(t7_1))
        print('t8:{0}'.format(t8))
        print('t8_1:{0}'.format(t8_1))

    def test(self):
        t1 = torch.tensor([1.0, 2.0, 3.0])
        print('t1 dtype:{0}; device:{1}; layout:{2}'.format(t1.dtype, t1.device, t1.layout))
        t2 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=torch.device('cuda:0'))
        print(t2)
        