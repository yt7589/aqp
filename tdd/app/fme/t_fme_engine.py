import unittest
import numpy as np
from app.fme.fme_engine import FmeEngine

class TFmeEngine(unittest.TestCase):
    def test_startup(self):
        fme_engine = FmeEngine()
        fme_engine.startup()

    def test_libra(self):
        # 1 Libra币兑换法币汇率
        libra_lcs = np.array([1.0174, 0.38671, 11.900, 0.085946, 0.58252])
        # 1美元兑换其他法币汇率
        dollar_lcs = np.array([6.89770, 1.06230, 114.38500, 1.26210, 1.0])
        # 1其他法币兑换美元
        lcs_dollar = 1 / dollar_lcs
        print(lcs_dollar)
        # 计算1 Libra兑换的美元数
        sum = 0.0
        for i in range(5):
            sum += libra_lcs[i]*lcs_dollar[i]
        print('1 Libra兑换美元：{0}'.format(sum))
        # 美元升值
        dollar_lcs[0] = 7.5
        lcs_dollar = 1 / dollar_lcs
        print(dollar_lcs)
        sum = 0.0
        for i in range(5):
            sum += libra_lcs[i]*lcs_dollar[i]
        print('1 Libra兑换美元：{0}'.format(sum))