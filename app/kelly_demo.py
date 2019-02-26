import numpy as np

'''
Kelly公式验试程序
'''

class KellyDemo(object):
    def __init__(self):
        self.name = 'KellyDemo'

    def startup(self):
        np.random.seed(1018)
        n = 10000
        head_prob = 0.5
        #tail_prob = 0.5
        sum = 0
        user_money = 100
        bid_ratio = 1.0
        bid_money = 0
        head_ratio = 10
        tail_ratio = 1
        '''
        guess = 1
        bid_ratio = head_ratio
        '''
        guess = 0
        bid_ratio = tail_ratio
        for i in range(n):
            bid_money = user_money / bid_ratio
            rn = np.random.random()
            if rn < head_prob:
                #print('正面')
                sum += 1
                if 1==guess:
                    user_money += bid_money * head_ratio
                else:
                    user_money -= bid_money * tail_ratio
            else:
                print('反面')
                if 1==guess:
                    user_money -= bid_money * head_ratio
                else:
                    user_money += bid_money * tail_ratio
            print('rount_{0}:押注：{2} => {1}'.format(i+1, user_money, bid_money))
            if user_money <= 10 or user_money>1100000:
                break
        print('正面次数：{0}; 钱：{1}'.format(sum, user_money))