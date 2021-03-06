#
import math
import numpy as np

class HoltWinters(object):
    def __init__(self):
        self.name = 'HoltWinters'
        
    def startup(self):
        self.predict_by_mean()
        
    def predict_by_mean(self):
        y = np.array([3,10,12,13,12,10,12], dtype=np.float32)
        return np.mean(y)
        
    def weighted_average(self, series, weights):
        mean = 0.0
        weights.reverse()
        for i in range(len(weights)):
            mean += series[-i-1] * weights[i]
        return mean
        
    def expopential_smoothing(self, series, alpha):
        result = [series[0]]
        for t in range(1, len(series)):
            result.append(alpha * series[t] + (1 - alpha) * result[t-1])
        return result
        
    def double_exponential_smoothing(self, series, alpha, beta):
        result = [series[0]]
        for n in range(1, len(series)+1):
            if n == 1:
                level, trend = series[0], series[1] - series[0]
            if n >= len(series): # we are forecasting
              value = result[-1]
            else:
              value = series[n]
            last_level, level = level, alpha*value + (1-alpha)*(level+trend)
            trend = beta*(level-last_level) + (1-beta)*trend
            result.append(level+trend)
        return result
        
    def initial_trend(self, series, slen):
        sum = 0.0
        for i in range(slen):
            sum += float(series[i+slen] - series[i]) / slen
        return sum / slen
        
    def initial_seasonal_components(self, series, slen):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(series)/slen)
        # compute season averages
        for j in range(n_seasons):
            season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
        # compute initial values
        for i in range(slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
            seasonals[i] = sum_of_vals_over_avg/n_seasons
        return seasonals
        
    def triple_exponential_smoothing(self, series, slen, alpha, beta, gamma, n_preds):
        result = []
        seasonals = self.initial_seasonal_components(series, slen)
        for i in range(len(series)+n_preds):
            if i == 0: # initial values
                smooth = series[0]
                trend = self.initial_trend(series, slen)
                result.append(series[0])
                continue
            if i >= len(series): # we are forecasting
                m = i - len(series) + 1
                result.append((smooth + m*trend) + seasonals[i%slen])
            else:
                val = series[i]
                last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
                trend = beta * (smooth-last_smooth) + (1-beta)*trend
                seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
                result.append(smooth+trend+seasonals[i%slen])
        return result