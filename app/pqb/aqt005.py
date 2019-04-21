from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from pykalman import KalmanFilter
import pylab as pl

class Aqt005(object):
    def __init__(self):
        self.name = 'Aqt005'
        
    def startup(self):
        print('卡尔曼滤波确定动态对冲比例')
        self.hedge_ratio()
        #self.kalman_filter()
        
    def hedge_ratio(self):
        etfs = ['TLT', 'IEI']
        start_date = "2010-8-01"
        end_date = "2016-08-01"
        # Obtain the adjusted closing prices from Yahoo finance
        '''
        etf_df1 = pdr.get_data_yahoo(etfs[0], start_date, end_date)
        etf_df2 = pdr.get_data_yahoo(etfs[1], start_date, end_date)
        prices = pd.DataFrame(index=etf_df1.index)
        prices[etfs[0]] = etf_df1["Adj Close"]
        prices[etfs[1]] = etf_df2["Adj Close"]
        prices.to_csv('./work/aqt005_001.txt', encoding='utf-8')
        '''
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
        prices = pd.read_csv('./work/aqt005_001.txt', encoding='utf-8', parse_dates=['Date'], date_parser=dateparse, index_col='Date')
        # 画散点图
        self.draw_date_coloured_scatterplot(etfs, prices)
        state_means, state_covs = self.calc_slope_intercept_kalman(etfs, prices)
        self.draw_slope_intercept_changes(prices, state_means)
        
    def draw_date_coloured_scatterplot(self, etfs, prices):
        """
        Create a scatterplot of the two ETF prices, which is
        coloured by the date of the price to indicate the
        changing relationship between the sets of prices
        """
        # Create a yellow-to-red colourmap where yellow indicates
        # early dates and red indicates later dates
        plen = len(prices)
        colour_map = plt.cm.get_cmap('YlOrRd')
        colours = np.linspace(0.1, 1, plen)
        # Create the scatterplot object
        scatterplot = plt.scatter(
        prices[etfs[0]], prices[etfs[1]],
        s=30, c=colours, cmap=colour_map,
        edgecolor='k', alpha=0.8
        )
        # Add a colour bar for the date colouring and set the
        # corresponding axis tick labels to equal string-formatted dates
        colourbar = plt.colorbar(scatterplot)
        colourbar.ax.set_yticklabels(
        [str(p.date()) for p in prices[::plen//9].index]
        )
        plt.xlabel(prices.columns[0])
        plt.ylabel(prices.columns[1])
        plt.show()
        
    def calc_slope_intercept_kalman(self, etfs, prices):
        """
        Utilise the Kalman Filter from the PyKalman package
        to calculate the slope and intercept of the regressed
        ETF prices.
        """
        delta = 1e-5
        mu0 = np.zeros(2)
        sigma0 = np.ones((2, 2))
        Q = delta / (1 - delta) * np.eye(2)
        At = np.eye(2)
        Ct = np.vstack(
            [prices[etfs[0]], np.ones(prices[etfs[0]].shape)]
        ).T[:, np.newaxis]
        R = 1.0
        kf = KalmanFilter(
            n_dim_obs=1,
            n_dim_state=2,
            initial_state_mean=mu0,
            initial_state_covariance=sigma0,
            transition_matrices=At,
            observation_matrices=Ct,
            observation_covariance=R,
            transition_covariance=Q
        )
        
        yt = prices[etfs[1]].values
        #state_means, state_covs = kf.em(observations).filter(observations)
        xt_means, xt_covs = kf.em(yt).filter(yt)
        return xt_means, xt_covs
        
    def draw_slope_intercept_changes(self, prices, state_means):
        """
        Plot the slope and intercept changes from the
        Kalman Filter calculated values.
        """
        pd.DataFrame(
            dict(
                slope=state_means[:, 0],
                intercept=state_means[:, 1]
            ), 
            index=prices.index
        ).plot(subplots=True)
        plt.show()
        
        
    def kalman_filter(self):
        random_state = np.random.RandomState(0)
        transition_matrix = [[1, 0.1], [0, 1]]
        transition_offset = [-0.1, 0.1]
        observation_matrix = np.eye(2) + random_state.randn(2, 2) * 0.1
        observation_offset = [1.0, -1.0]
        transition_covariance = np.eye(2)
        observation_covariance = np.eye(2) + random_state.randn(2, 2) * 0.1
        initial_state_mean = [5, -5]
        initial_state_covariance = [[1, 0.1], [-0.1, 1]]
        # sample from model
        kf = KalmanFilter(
            transition_matrix, observation_matrix, transition_covariance,
            observation_covariance, transition_offset, observation_offset,
            initial_state_mean, initial_state_covariance,
            random_state=random_state
        )
        
        print('o:{0}; s:{1}!'.format(kf.n_dim_obs, kf.n_dim_state))
        
        states, observations = kf.sample(
            n_timesteps=50,
            initial_state=initial_state_mean
        )
        filtered_state_estimates, filtered_state_covariances = kf.filter(observations)
        plt.plot(filtered_state_estimates)
        plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        