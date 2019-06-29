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
        self.max_min_file = './work/btc_max_min.csv'
        self.bst = None
        self.fme_dataset = FmeDataset()
        self.X, self.y = self.fme_dataset.load_bitcoin_dataset()
        self.model = self.train_baby_agent()
        self.df = None
        self.fme_env = None
        self.max_min_file = './work/btc_max_min.csv'
        self.dataset_size = 10
        self.cached_quotation = np.loadtxt(self.max_min_file, delimiter=',')
        # predict example
        '''
        x1 = np.array([self.X[0]])
        xg1 = xgb.DMatrix( x1, label=x1)
        pred = self.model.predict( xg1 )
        '''

    def normalize_ds(self, x, ds_max, ds_min):
        for i in range(25):
            x[i] = (x[i] - ds_min[i%5]) / (ds_max[i%5] - ds_min[i%5])


    def choose_action(self, idx, obs):
        '''  '''
        commission = self.fme_env.commission
        frame_size = self.fme_dataset.frame_size
        recs = self.df.iloc[idx-frame_size+1:idx+1]
        datas = np.array(recs)
        ds = datas[:, 3:8]
        print('ds.shape:{0}; frame_size={1}; idx={2}'.format(ds.shape, frame_size, idx))
        ds = np.reshape(ds, (frame_size*5, ))
        date_quotation = ds[20:25]
        if self.fme_env.btc_held <= 0.00000001:
            x = np.append(ds, [0.0])
        else:
            x = np.append(ds, [1.0])
        self.add_quotation_tick(self.cached_quotation, [x[20], x[21], x[22], x[23], x[24]])
        ds_max = np.amax(self.cached_quotation, axis=0)
        ds_min = np.amin(self.cached_quotation, axis=0)
        self.normalize_ds(x, ds_max, ds_min)
        print('x:{0:04f}, {1:04f}, {2:04f}, {3:04f}, {4:04f}, {5:04f}, '
                '{6:04f}, {7:04f}, {8:04f}, {9:04f}, {10:04f}, {11:04f},'
                '{12:04f}, {13:04f}, {14:04f}, {15:04f}, {16:04f}, {17:04f}, {18:04f},'
                '{19:04f}, {20:04f}, {21:04f}, {22:04f}, {23:04f}, {24:04f}, {25:04f},'.format(
                    x[0], x[1], x[2], x[3], x[4], 
                    x[5], x[6], x[7], x[8], x[9],
                    x[10], x[11], x[12], x[13], x[14],
                    x[15], x[16], x[17], x[18], x[19],
                    x[20], x[21], x[22], x[23], x[24], x[25]
                ))
        xg = xgb.DMatrix([x], label=x)
        pred = self.model.predict(xg)
        action_type = np.argmax(pred)
        print('pred:{0}; [{1:02f}, {2:02f}, {3:02f}]=>{4}'.format(
            pred.shape, pred[0][0], pred[0][1], pred[0][2],
            np.argmax(pred[0]))
        )
        if 0 == action_type:
            action = np.array([0, 10])
        elif 1 == action_type:
            action = np.array([1, 10])
        else:
            action = np.array([2, 10])
        self.x = x
        self.action = action_type
        return action

    def train_drl_agent(self, weight):
        # 将x添加到self.X_train中，将action添加到self.y_train中
        self.X_train = np.delete(self.X_train, 0, axis=0)
        self.X_train = np.append(self.X_train, [self.x], axis=0)
        self.y_train = np.delete(self.y_train, 0, axis=0)
        self.y_train = np.append(self.y_train, [self.action], axis=0)
        # 计算执行action后的净值与之前的净值比值作为本样本的权重，
        # 更新self.rlw的权重值
        self.rlw = np.delete(self.rlw, 0, axis=0)
        self.rlw = np.append(self.rlw, [weight], axis=0)
        # 重新对训练模型k遍
        self.train_model(num_round=1)
    
    def train_model(self, num_round = 1):
        X_validation, y_validation = self.X_train, self.y_train
        X_test, y_test = self.X_train, self.y_train
        xg_train = xgb.DMatrix(self.X_train, label=self.y_train, weight=self.rlw)
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
        if self.bst is None and os.path.exists(self.model_file):
            print('load xgboost model...')
            self.bst = xgb.Booster({})
            self.bst.load_model(self.model_file)
        else:
            print('########## build xgboost model...')
            self.bst = xgb.train(xgb_params, xg_train, num_round, watchlist )
            self.bst.save_model(self.model_file)
        return self.bst

    def train_baby_agent(self):
        ''' 在这里仅进行初步训练，得到一个基本可用的模型 '''
        fme_dataset = FmeDataset()
        self.X_train, self.y_train = fme_dataset.load_bitcoin_dataset()
        self.rlw = np.ones((self.X_train.shape[0])) # 样本权重
        X_validation, y_validation = self.X_train, self.y_train
        X_test, y_test = self.X_train, self.y_train
        bst = self.train_model(num_round=2000)

        x1 = np.array([X_test[0]])
        xg1 = xgb.DMatrix( x1, label=x1)
        pred = bst.predict( xg1 )
        print('x1:{0}'.format(x1))
        print('pred:{0}; [{1:02f}, {2:02f}, {3:02f}]=>{4}'.format(
            pred.shape, pred[0][0], pred[0][1], pred[0][2],
            np.argmax(pred[0]))
        )
        #print ('predicting, classification error=%f' % (sum( int(pred[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test)) ))
        plot_importance(bst)
        plt.show()
        return bst
        
    def add_quotation_tick(self, cached_quotation, tick):
        ''' 将当天的行情加入到历史行情列表中，用于计算各变量的最大最小值 '''
        if cached_quotation.shape[0]<self.dataset_size:
            cached_quotation = np.append(cached_quotation, [tick], axis=0)
        else:
            # 删除最前面条目
            cached_quotation = np.delete(cached_quotation, 0, axis=0)
            cached_quotation = np.append(cached_quotation, [tick], axis=0)
        return cached_quotation