import csv
import matplotlib.pyplot as plt

class QhEvaluator(object):
    def __init__(self):
        self.name = 'QhEvaluator'
    
    def draw_cumulative_returns(self):
        '''
        绘制收益率曲线
        @version v0.0.1 闫涛 2019-04-04
        '''
        with open('out/tradelog_2019-04-04.csv', 'r', newline='') as fd:
            rows = csv.reader(fd, delimiter=',', quotechar='|')
            prices = []
            crs = []
            x = []
            next(fd)
            for row in rows:
                if len(row) > 0 and 'SPY'==row[1] and 'BOT'==row[2]:
                    prices.append(float(row[5]))
                    
            for idx in range(1, len(prices)):
                crs.append((prices[idx] - prices[idx-1])/prices[idx-1] + 1)
                x.append(idx-1)
                #print('{0}: {1} - {2} / {3}'.format(idx, prices[idx], prices[idx-1], prices[idx-1]))
             
            for cr in crs:
                print('return: {0}!'.format(cr))
            
            fig, ax = plt.subplots()
            ax.plot(x, crs)
            plt.show()
            