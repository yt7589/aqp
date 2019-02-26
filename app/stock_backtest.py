import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class StockBacktest(object):
    def __init__(self):
        self.name = 'StockBacktest'

    def startup(self):
        print('股票回测研究平台 v0.0.1')
        # 选定股票


