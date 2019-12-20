#
import math
import numpy as np
import matplotlib.pyplot as plt

class Chp003C003(object):
    def __init__(self):
        self.name = ''

    def run(self):
        print('一元高斯分布图像')
        mu = 3.0
        sigma = 0.5
        x = np.linspace(-1.0, 6.0, 100)
        y = self.gaussian(x, mu, sigma)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title('一元高斯分布图像')
        plt.plot(x, y, '-b')
        plt.show()

    def gaussian(self, x, mu, sigma):
        v1 = 1 / (math.sqrt(2 * math.pi) * sigma)
        v2 = (x-mu)*(x-mu) / (2 * sigma * sigma)
        v3 = np.exp(-v2)
        return v1 * v3


if '__main__' == __name__:
    app = Chp003C003()
    app.run()