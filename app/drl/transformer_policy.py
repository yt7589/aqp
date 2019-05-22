#
import os 
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import numpy as np
import random
from sklearn.utils import shuffle
import random, os, sys
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
import tensorflow as tf
from tensorflow.python.keras.layers import Layer

try:
    from dataloader import TokenList, pad_to_longest
    # for transformer
except: pass
#
from app.drl.multi_head_attention import MultiHeadAttention
from app.drl.layer_normalization import LayerNormalization
from app.drl.positionwise_feed_forward import PositionwiseFeedForward
from app.drl.encoder_layer import EncoderLayer
from app.drl.scaled_dot_product_attention import ScaledDotProductAttention
from app.drl.transformer import Transformer

class TransformerPolicy(object):
    SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
    FUTURE_PERIOD_PREDICT = 1  # how far into the future are we trying to predict?
    RATIO_TO_PREDICT = "mid"
    EPOCHS = 10  # how many passes through our data
    BATCH_SIZE = 1024

    def __init__(self):
        self.data_file = './data/GE2.csv'
        self.model_file = './work/transformer_stock.drl'
        self.first_run = True

    def startup(self):
        self.df = pd.read_csv('./data/GE2.csv',delimiter=',',
            usecols=['Date','Open','High','Low','Close', 'Volume']
        )

        # Sort DataFrame by date
        self.df = self.df.sort_values('Date')
        self.df['mid'] = (self.df['Low'] + self.df['High'])/2.0
        #self.draw_mid_price(self.df)
        self.df['future'] = self.df[TransformerPolicy.RATIO_TO_PREDICT].shift(-TransformerPolicy.FUTURE_PERIOD_PREDICT)
        self.df['target'] = list(map(self.classify, self.df[TransformerPolicy.RATIO_TO_PREDICT], self.df['future']))
        
        self.X_train, self.y_train, \
                    self.X_valid, self.y_valid, \
                    self.X_test, self.y_test, \
                    self.X_train_2, self.y_train_2 = \
                    self.prepare_dataset(self.df)
        '''
        self.draw_target_dataset(self.X_train, self.y_train, 
                    self.X_valid, self.y_valid, 
                    self.X_test, self.y_test, 
                    self.X_train_2, self.y_train_2)
        '''
        self.train(self.X_train, self.y_train, 
                    self.X_valid, self.y_valid,
                    self.X_test)


        print('^_^ the end!')

    def train(self, X_train, y_train, X_valid, y_valid, X_test):
        X_train, y_train = shuffle(X_train, y_train)
        NAME = f"{TransformerPolicy.SEQ_LEN}-SEQ-{TransformerPolicy.FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"
        multi_head = self.build_model()
        #if not self.first_run:
         #   multi_head.load_weights(self.model_file)


        #multi_head.save_weights(self.model_file)

        
        print('tf.executing_eagerly()={0}; version={1}'.format(tf.executing_eagerly(), tf.__version__))
        
        multi_head.save(self.model_file)
        i_debug = 1
        if 1 == i_debug:
            return
        

        multi_head.summary()
        multi_head.fit(X_train, y_train,
                    batch_size=TransformerPolicy.BATCH_SIZE,
                    epochs=TransformerPolicy.EPOCHS,
                    validation_data=(X_valid, y_valid), 
                    #callbacks = [checkpoint , lr_reduce]
             )
        # save weights
        #multi_head.save_weights(self.model_file)
        predicted_stock_price_multi_head = multi_head.predict(X_test)
        predicted_stock_price_multi_head = np.vstack((np.full((60,1), np.nan), predicted_stock_price_multi_head))

        plt.figure(figsize = (18,9))
        plt.plot(self.test_data, color = 'black', label = 'GE Stock Price')
        plt.plot(predicted_stock_price_multi_head, color = 'green', label = 'Predicted GE Mid Price')
        plt.title('GE Mid Price Prediction', fontsize=30)
        #plt.xticks(range(0,df.shape[0],50),df['Date'].loc[::50],rotation=45)
        plt.xlabel('Date')
        plt.ylabel('GE Mid Price')
        plt.legend(fontsize=18)
        plt.show()






    def build_model(self):
        inp = Input(shape = (TransformerPolicy.SEQ_LEN, 1))
        
        # LSTM before attention layers
        x = Bidirectional(LSTM(128, return_sequences=True))(inp)
        x = Bidirectional(LSTM(64, return_sequences=True))(x) 
            
        x, slf_attn = MultiHeadAttention(n_head=3, d_model=300, d_k=64, d_v=64, dropout=0.1)(x, x, x)
            
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        conc = Dense(64, activation="relu")(conc)
        x = Dense(1, activation="sigmoid")(conc)      

        model = Model(inputs = inp, outputs = x)
        model.compile(
            loss = "mean_squared_error", 
            #optimizer = Adam(lr = config["lr"], decay = config["lr_d"]), 
            optimizer = "adam")
        
        # Save entire model to a HDF5 file
        #model.save('./work/transformer_stock.h5')
        
        return model

    def draw_mid_price(self, df):
        # Double check the result
        print(df.head())
        plt.figure(figsize = (12,6))
        plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
        plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
        plt.xlabel('Date',fontsize=18)
        plt.ylabel('Mid Price',fontsize=18)
        plt.show()

    def classify(self, current, future):
        if float(future) > float(current):
            return 1
        else:
            return 0

    def prepare_dataset(self, df):
        times = sorted(df.index.values)
        last_10pct = sorted(df.index.values)[-int(0.1*len(times))]
        last_20pct = sorted(df.index.values)[-int(0.2*len(times))]
        test_df = df[(df.index >= last_10pct)]
        validation_df = df[(df.index >= last_20pct) & (df.index < last_10pct)]
        train_df = df[(df.index < last_20pct)]
        #
        train_df.drop(columns=["Date", "future", 'Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
        validation_df.drop(columns=["Date", "future", 'Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
        test_df.drop(columns=["Date", "future", 'Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)# don't need this anymore.

        train_data = train_df[TransformerPolicy.RATIO_TO_PREDICT].as_matrix()
        valid_data = validation_df[TransformerPolicy.RATIO_TO_PREDICT].as_matrix()
        self.test_data = test_df[TransformerPolicy.RATIO_TO_PREDICT].as_matrix()
        train_data = train_data.reshape(-1,1)
        valid_data = valid_data.reshape(-1,1)
        self.test_data = self.test_data.reshape(-1,1)
        #
        scaler = MinMaxScaler()
        smoothing_window_size = 2500
        for di in range(0,10000,smoothing_window_size):
            scaler.fit(train_data[di:di+smoothing_window_size,:])
            train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

        # You normalize the last bit of remaining data
        scaler.fit(train_data[di+smoothing_window_size:,:])
        train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

        # Reshape both train and test data
        train_data = train_data.reshape(-1)

        # Normalize test data and validation data
        valid_data = scaler.transform(valid_data).reshape(-1)
        self.test_data = scaler.transform(self.test_data).reshape(-1)

        # Now perform exponential moving average smoothing
        # So the data will have a smoother curve than the original ragged data
        EMA = 0.0
        gamma = 0.1
        for ti in range(11000):
            EMA = gamma*train_data[ti] + (1-gamma)*EMA
            train_data[ti] = EMA

        # Used for visualization and test purposes
        all_mid_data = np.concatenate([train_data,valid_data, self.test_data],axis=0)

        X_train = []
        y_train = []
        for i in range(TransformerPolicy.SEQ_LEN, len(train_data)):
            X_train.append(train_data[i-TransformerPolicy.SEQ_LEN:i])
            y_train.append(train_data[i + (TransformerPolicy.FUTURE_PERIOD_PREDICT-1)])
        X_train, y_train = np.array(X_train), np.array(y_train)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        X_valid = []
        y_valid = []
        for i in range(TransformerPolicy.SEQ_LEN, len(valid_data)):
            X_valid.append(valid_data[i-TransformerPolicy.SEQ_LEN:i])
            y_valid.append(valid_data[i+(TransformerPolicy.FUTURE_PERIOD_PREDICT-1)])
        X_valid, y_valid = np.array(X_valid), np.array(y_valid)

        X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))

        X_test = []
        y_test = []
        for i in range(TransformerPolicy.SEQ_LEN, len(self.test_data)):
            X_test.append(self.test_data[i-TransformerPolicy.SEQ_LEN:i])
            y_test.append(self.test_data[i+(TransformerPolicy.FUTURE_PERIOD_PREDICT-1)])
            
        X_test, y_test = np.array(X_test), np.array(y_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        X_train_2 = []
        y_train_2 = []
        for i in range(TransformerPolicy.SEQ_LEN, len(train_data)):
            X_train_2.append(train_data[i-TransformerPolicy.SEQ_LEN:i])
            y_train_2.append(train_data[i + (TransformerPolicy.FUTURE_PERIOD_PREDICT-1)])
        X_train_2, y_train_2 = np.array(X_train_2), np.array(y_train_2)

        X_train_2 = np.reshape(X_train_2, (X_train_2.shape[0], X_train_2.shape[1], 1))


        return X_train, y_train, X_valid, \
                y_valid, X_test, y_test, \
                X_train_2, y_train_2

    def draw_target_dataset(self, X_train, y_train, 
                X_valid, y_valid, X_test, y_test, 
                X_train_2, y_train_2):
        plt.figure(figsize=(15, 5))

        plt.plot(np.arange(y_train_2.shape[0]), y_train_2, color='blue', label='train target')

        plt.plot(np.arange(y_train_2.shape[0], y_train_2.shape[0]+y_valid.shape[0]), y_valid,
                color='gray', label='valid target')

        plt.plot(np.arange(y_train_2.shape[0]+y_valid.shape[0],
                        y_train_2.shape[0]+y_valid.shape[0]+y_test.shape[0]),
                y_test, color='black', label='test target')


        plt.title('Séparation des données')
        plt.xlabel('time [days]')
        plt.ylabel('normalized price')
        plt.legend(loc='best')
        plt.show()
