#
import numpy as np
import matplotlib.pyplot as plt

class Chp003C001(object):
    def __init__(self):
        self.name = 'Chp003C001'

    def run(self):
        print('绘制sigmoid函数曲线')
        x = np.linspace(-10.0, 10.0, 500)
        y = 1 / (1 + np.exp(-x))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title('sigmoid函数图像')
        plt.plot(x, y, '-b')
        plt.show()

if '__main__' == __name__:
    app = Chp003C001()
    app.run()