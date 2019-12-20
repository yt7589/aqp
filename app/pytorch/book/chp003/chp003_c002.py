#
import numpy as np
import matplotlib.pyplot as plt

class Chp003C002(object):
    def __init__(self):
        self.name = ''

    def run(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title('牛顿法求解示意')
        self.draw_curve()
        self.draw_x0()
        self.draw_x1()
        plt.show()
    
    def draw_curve(self):
        # 绘制x轴
        xx = np.array([0.0, 5.0])
        xy = np.array([0.0, 0.0])
        plt.plot(xx, xy, '-g')
        # 绘制曲线
        x = np.linspace(0, 5, 100)
        y = x*x - 2.25
        plt.plot(x, y, '-b')
        plt.annotate(s=r'目标(1.5)',xy=(1.5, 0.0),\
            xytext=(1.0,2.0),weight='bold',color='black',\
            arrowprops=dict(arrowstyle='-|>',\
            connectionstyle='arc3',color='blue'),\
            bbox=dict(boxstyle='round,pad=0.5', fc='white', \
            ec='k',lw=1 ,alpha=0.4))

    def draw_x0(self):
        # x0点垂线
        x00x = np.array([4.5, 4.5])
        x00y = np.array([0.0, 4.5*4.5-2.25])
        plt.plot(x00x, x00y, '-r')
        plt.annotate(s=r'x0',xy=(4.5, 4.5*4.5-2.25),
            xytext=(4.0,18.0),weight='bold',color='black',\
            arrowprops=dict(arrowstyle='-|>',\
            connectionstyle='arc3',color='blue'),\
            bbox=dict(boxstyle='round,pad=0.5', fc='white', \
            ec='k',lw=1 ,alpha=0.4))
        plt.text(4.4, -1.0, '4.5')
        # x0点处切线并与x轴相交
        x01x = np.array([4.5, 2.5])
        x01y = np.array([4.5*4.5-2.25, 0])
        plt.plot(x01x, x01y, '-r')
        plt.text(2.4, -1.0, '2.5')

    def draw_x1(self):
        # x1点垂线
        x10x = np.array([2.5, 2.5])
        x10y = np.array([0.0, 2.5*2.5-2.25])
        plt.plot(x10x, x10y, '-r')
        plt.annotate(s=r'x1',xy=(2.5, 2.5*2.5-2.25),\
            xytext=(2.0,6.0),weight='bold',color='black',\
            arrowprops=dict(arrowstyle='-|>',\
            connectionstyle='arc3',color='blue'),\
            bbox=dict(boxstyle='round,pad=0.5', fc='white', \
            ec='k',lw=1 ,alpha=0.4))
        # x1点切线
        x11x = np.array([2.5, 1.7])
        x11y = np.array([2.5*2.5-2.25, 0])
        plt.plot(x11x, x11y, '-r')
        plt.text(1.6, -1.0, '1.7')

if '__main__' == __name__:
    app = Chp003C002()
    app.run()
