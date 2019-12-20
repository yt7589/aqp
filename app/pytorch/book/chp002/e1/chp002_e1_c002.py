#
import numpy as np
import matplotlib.pyplot as plt

def draw_squre_curve():
    x = np.linspace(0, 3, 200)
    y = (x - 1.5) * (x - 1.5)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('二次曲线梯度下降算法示意图')
    plt.plot(x, y, '-b')
    x0 = np.array([0.0, 3.0])
    y0 = np.array([0.0, 0.0])
    plt.plot(x0, y0, '-b')
    x1 = np.array([2.5, 2.5])
    y1 = np.array([0.0, 1])
    plt.plot(x1, y1, '-r')
    x2 = np.array([2.0, 2.0])
    y2 = np.array([0.0, 0.25])
    plt.plot(x2, y2, '-r')
    x3 = np.array([0.0, 0.0])
    y3 = np.array([0.0, 2.25])
    plt.plot(x3, y3, '-r')
    plt.show()

def main():
    print('绘制二次曲线')
    draw_squre_curve()

if '__main__' == __name__:
    main()