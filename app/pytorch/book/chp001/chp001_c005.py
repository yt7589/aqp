#
import numpy as np
import torch

class Chp001C005(object):
    def __init__(self):
        self.name = 'app.pytorch.book.chp001.Chp001C005'

    def run(self):
        t1 = torch.tensor([
            [1.1, 1.2, 1.3],
            [2.1, 2.2, 2.3]
        ])
        t2 = torch.tensor([
            [10.1, 11.0, 12.0],
            [20.0, 21.0, 22.0]
        ])
        print('操作符：\r\nt1+t2={0};\r\nt1-t2={1};\r\nt1*t2={2};'\
            '\r\nt1/t2={3}'\
            .format(t1+t2, t1-t2, t1*t2, t1/t2))
        print('函数：\r\nt1+t2={0};\r\nt1-t2={1};\r\nt1*t2={2};'\
            '\r\nt1/t2={3}'\
            .format(t1.add(t2), t1.sub(t2), t1.mul(t2), t1.div(t2)))
        print('张量加标量（操作符形式）：\r\nt1+t2={0};\r\n'\
            't1-t2={1};\r\nt1*t2={2};\r\nt1/t2={3}'\
            .format(t1+2, t1-2, t1*2, t1/2))
        print('逻辑运算符：\r\nt1>1.2{0};\r\n'\
            't1<1.2{1};\r\nt1==1.1{2}'\
            .format(t1>1.2, t1<1.2, t1==1.1))
        t3 = torch.tensor([100.0, 200.0, 300.0])
        print('不同形状标量运算：\r\nt1+t3={0};\r\nt1-t3={1};\r\n'\
            't1*t3={2};\r\nt1/t3={3}'\
            .format(t1+t3, t1-t3, t1*t3, t1/t3))
        print('以张量为参数函数：{0}'.format(self.fx(t1)))

    def fx(self, x):
        return x*100 + 8
