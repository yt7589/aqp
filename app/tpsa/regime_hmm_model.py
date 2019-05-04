#
from __future__ import print_function

import datetime
import pickle
import warnings

import hmmlearn.hmm as hmm
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator
import numpy as np
import pandas as pd
import seaborn as sns

class RegimeHmmModel(object):
    def __init__(self):
        self.name = 'RegimeHmmModel'
        self.data_file = './data/ICBC.csv' # 数据文件
        self.model_file = './work/hmm.pkl' # HMM模型文件
        
    def train(self):
        # 禁止sklearn的过期警告信息
        warnings.filterwarnings("ignore")
        csv_filepath = './data/ICBC.csv'
        end_date = datetime.datetime(2018, 12, 31)
        df = self.get_prices_df(csv_filepath, end_date)
        rets = np.column_stack([df['Returns']])
        print(rets)
        hmm_model = hmm.GaussianHMM(
            n_components=2, covariance_type="full", n_iter=1000
        ).fit(rets)
        print("Model Score:", hmm_model.score(rets))
        hidden_states = hmm_model.predict(rets)
        self.plot_in_sample_hidden_states(hmm_model, df, hidden_states)
        # 将模型保存为文件（因为hmm每次运行出来的state=0,1是不固定的）
        pickle.dump(hmm_model, open(self.model_file, "wb"))
        return hmm_model
        
    def get_prices_df(self, csv_filepath, end_date):
        """
        Obtain the prices DataFrame from the CSV file,
        filter by the end date and calculate the 
        percentage returns.
        """
        df = pd.read_csv(
            csv_filepath, header=0,
            names=[
                "Date", "Open", "High", "Low", 
                "Close", "Volume", "Adj Close"
            ],
            index_col="Date", parse_dates=True
        )
        df["Returns"] = df["Adj Close"].pct_change()
        df = df[:end_date.strftime("%Y-%m-%d")]
        df.dropna(inplace=True)
        return df
        
        
    def plot_in_sample_hidden_states(self, hmm_model, df, hidden_states):
        """
        Plot the adjusted closing prices masked by 
        the in-sample hidden states as a mechanism
        to understand the market regimes.
        """
        # Create the correctly formatted plot
        fig, axs = plt.subplots(
            hmm_model.n_components, 
            sharex=True, sharey=True
        )
        colours = cm.rainbow(
            np.linspace(0, 1, hmm_model.n_components)
        )
        for i, (ax, colour) in enumerate(zip(axs, colours)):
            mask = hidden_states == i
            ax.plot_date(
                df.index[mask], 
                df["Adj Close"][mask], 
                ".", linestyle='none', 
                c=colour
            )
            ax.set_title("Hidden State #%s" % i)
            ax.xaxis.set_major_locator(YearLocator())
            ax.xaxis.set_minor_locator(MonthLocator())
            ax.grid(True)
        plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    