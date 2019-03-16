import numpy as np
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from util.app_util import AppUtil
from matplotlib.font_manager import FontProperties

class CapitalCurve(object):
    @staticmethod
    def draw_curve_demo():
        dates = []
        idx = 0.1
        today = AppUtil.get_today_obj()
        curr_date = AppUtil.parse_date('20190101')
        ys = []
        mu = 0
        sigma = 10
        num = 100
        rand_data = np.random.normal(mu, sigma, num)
        print(rand_data)
        i = 0
        while curr_date <= today:
            dates.append(AppUtil.format_date(curr_date, AppUtil.DF_HYPHEN))
            curr_date += timedelta(days=1)
            ys.append(idx*idx + 100 + rand_data[i])
            idx += 0.1
            i += 1

        xs = [datetime.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]
        font = FontProperties(fname='./work/simsun.ttc') # 载入中文字体
        plt.rcParams['axes.unicode_minus']=False # 正确显示负号
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=(mdates.MO)))
        plt.title('总资产变化曲线', fontproperties=font)
        plt.xlabel('日期', fontproperties=font)
        plt.ylabel('总资产（万元）' , fontproperties=font)
        plt.plot(xs, ys)
        plt.gcf().autofmt_xdate() 
        plt.show()
