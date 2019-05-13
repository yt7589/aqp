import unittest
import matplotlib.pyplot as plt
from app.drl.holt_winters import HoltWinters

class THoltWinters(unittest.TestCase):
    def test_weighted_average(self):
        holt_winters = HoltWinters()
        series = [3.0, 10.0, 12.0, 13.0, 12.0, 10.0, 12.0]
        weights = [0.1, 0.2, 0.3, 0.4]
        mean = holt_winters.weighted_average(series, weights)
        print('mean={0}'.format(mean))
        self.assertTrue(True)
        
    def test_expopential_smoothing(self):
        series = [3.0, 10.0, 12.0, 13.0, 12.0, 10.0, 12.0]
        alpha = 0.9
        holt_winters = HoltWinters()
        result = holt_winters.expopential_smoothing(series, alpha)
        print(result)
        self.assertTrue(True)
        
    def test_expopential_smoothing_2(self):
        series = [3.0, 10.0, 12.0, 13.0, 12.0, 10.0, 12.0]
        alpha = 0.1
        holt_winters = HoltWinters()
        result = holt_winters.expopential_smoothing(series, alpha)
        print(result)
        self.assertTrue(True)
        
    def test_expopential_smoothing_3(self):
        holt_winters = HoltWinters()
        series = [3.0, 10.0, 12.0, 13.0, 12.0, 10.0, 12.0]
        alpha = 0.1
        r1 = holt_winters.expopential_smoothing(series, alpha)
        alpha = 0.9
        r2 = holt_winters.expopential_smoothing(series, alpha)
        
        fig, ax = plt.subplots()
        
        ax.plot(series, label='series')
        ax.plot(r1, label='a=0.1')
        ax.plot(r2, label='a=0.9')
        
        legend = ax.legend(loc='lower right', shadow=True, fontsize='medium')
        plt.show()
        
    def test_double_expopential_smoothing(self):
        holt_winters = HoltWinters()
        series = [3.0, 10.0, 12.0, 13.0, 12.0, 10.0, 12.0]
        alpha = 0.9
        beta = 0.9
        result = holt_winters.double_exponential_smoothing(series, alpha, beta)
        
        fig, ax = plt.subplots()
        
        ax.plot(series, label='series')
        ax.plot(result, label='predict')
        
        legend = ax.legend(loc='lower right', shadow=True, fontsize='medium')
        plt.show()
        
    def test_initial_trend(self):
        series = [30,21,29,31,40,48,53,47,37,39,31,29,17,9,20,24,27,35,41,38,
          27,31,27,26,21,13,21,18,33,35,40,36,22,24,21,20,17,14,17,19,
          26,29,40,31,20,24,18,26,17,9,17,21,28,32,46,33,23,28,22,27,
          18,8,17,21,31,34,44,38,31,30,26,32]
        holt_winters = HoltWinters()
        L = 12
        result = holt_winters.initial_trend(series, L)
        print(result) # 正确结果：-0.7847222222222222
        fig, ax = plt.subplots()
        
        ax.plot(series, label='series')
        
        legend = ax.legend(loc='lower right', shadow=True, fontsize='medium')
        plt.show()
        self.assertTrue(True)
        
    def test_initial_seasonal_components(self):
        series = [30,21,29,31,40,48,53,47,37,39,31,29,17,9,20,24,27,35,41,38,
          27,31,27,26,21,13,21,18,33,35,40,36,22,24,21,20,17,14,17,19,
          26,29,40,31,20,24,18,26,17,9,17,21,28,32,46,33,23,28,22,27,
          18,8,17,21,31,34,44,38,31,30,26,32]
        holt_winters = HoltWinters()
        L = 12
        result = holt_winters.initial_seasonal_components(series, L)
        print(result) # 正确结果：{0: -7.4305555555555545, 1: -15.097222222222221, 2: -7.263888888888888, 3: -5.097222222222222, 4: 3.402777777777778, 5: 8.069444444444445, 6: 16.569444444444446, 7: 9.736111111111112, 8: -0.7638888888888887, 9: 1.902777777777778, 10: -3.263888888888889, 11: -0.7638888888888887}
        fig, ax = plt.subplots()
        
        ax.plot(series, label='series')
        
        legend = ax.legend(loc='lower right', shadow=True, fontsize='medium')
        plt.show()
        self.assertTrue(True)
        
    def test_triple_exponential_smoothing(self):
        series = [30,21,29,31,40,48,53,47,37,39,31,29,17,9,20,24,27,35,41,38,
          27,31,27,26,21,13,21,18,33,35,40,36,22,24,21,20,17,14,17,19,
          26,29,40,31,20,24,18,26,17,9,17,21,28,32,46,33,23,28,22,27,
          18,8,17,21,31,34,44,38,31,30,26,32]
        holt_winters = HoltWinters()
        L = 12
        alpha = 0.716
        beta = 0.029
        gamma = 0.993
        n_preds = 24
        result = holt_winters.triple_exponential_smoothing(series, L, alpha, beta, gamma, n_preds)
        
        fig, ax = plt.subplots()
        ax.plot(result, label='predict')
        ax.plot(series, label='series')
        legend = ax.legend(loc='lower right', shadow=True, fontsize='medium')
        plt.show()
        self.assertTrue(True)
        
        