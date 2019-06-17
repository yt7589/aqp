import os
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from app.fme.fme_dataset import FmeDataset

class FmeXgbAgent(object):
    def __init__(self):
        self.name = 'FmeXgbAgent'
        self.model_file = './work/btc_drl.xgb'

    def train_baby_agent(self):
        ''' 在这里仅进行初步训练，得到一个基本可用的模型 '''
        fme_dataset = FmeDataset()
        X_train, y_train = fme_dataset.load_bitcoin_dataset()
        X_validation, y_validation = X_train, y_train
        X_test, y_test = X_train, y_train
        rlw = np.ones((X_train.shape[0])) # 样本权重
        xg_train = xgb.DMatrix(X_train, label=y_train, weight=rlw)
        xg_test = xgb.DMatrix( X_test, label=y_test)
        xgb_params = {
            'learning_rate': 0.1,  # 步长
            'n_estimators': 10,
            'max_depth': 5,  # 树的最大深度
            'objective': 'multi:softprob',
            'num_class': 3,
            # 决定最小叶子节点样本权重和，如果一个叶子节点的样本权重和小于
            # min_child_weight则拆分过程结束。
            'min_child_weight': 1, 
            # 指定了节点分裂所需的最小损失函数下降值。
            # 这个参数的值越大，算法越保守 
            'gamma': 0,  
            'silent': 0,  # 输出运行信息
            # 每个决策树所用的子样本占总样本的比例（作用于样本）
            'subsample': 0.8,  
            # 建立树时对特征随机采样的比例（作用于特征）典型值：0.5-1
            'colsample_bytree': 0.8,  
            'nthread': 4,
            'seed': 27
        }
        watchlist = [ (xg_train,'train'), (xg_test, 'test') ]
        num_round = 180
        if os.path.exists(self.model_file):
            print('load xgboost model...')
            bst = xgb.Booster({})
            bst.load_model(self.model_file)
        else:
            print('build xgboost model...')
            bst = xgb.train(xgb_params, xg_train, num_round, watchlist )
            bst.save_model(self.model_file)
        pred = bst.predict( xg_test )
        print('pred:{0}; {1}=>{2}'.format(pred.shape, pred[0], np.argmax(pred[0])))
        #print ('predicting, classification error=%f' % (sum( int(pred[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test)) ))
        plot_importance(bst)
        plt.show()