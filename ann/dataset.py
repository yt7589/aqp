# 对数据集进行处理的公共类

class Dataset(object):
    
    @staticmethod
    def normalize_data(datas, idx, mus, stds):
        ''' 归一化方法：减去均值再除以标准差 '''
        datas[:, idx:idx+1] = (datas[:, idx:idx+1] - mus[idx]) / stds[idx]
        
    @staticmethod
    def normalize_datas(datas, mus, stds):
        ''' 对开盘价、最高价、最低价、收盘价等进行归一化 '''
        Dataset.normalize_data(datas, 
                    Dataset.open_idx,
                    mus, stds
        )
        StockDailySvmModelEvaluator.normalize_data(datas, 
                    StockDailySvmModelEvaluator.high_idx,
                    mus, stds
        )
        StockDailySvmModelEvaluator.normalize_data(datas, 
                    StockDailySvmModelEvaluator.low_idx,
                    mus, stds
        )
        StockDailySvmModelEvaluator.normalize_data(datas, 
                    StockDailySvmModelEvaluator.close_idx,
                    mus, stds
        )
        StockDailySvmModelEvaluator.normalize_data(datas, 
                    StockDailySvmModelEvaluator.pre_close_idx,
                    mus, stds
        )
        StockDailySvmModelEvaluator.normalize_data(datas, 
                    StockDailySvmModelEvaluator.amt_chg_idx,
                    mus, stds
        )
        StockDailySvmModelEvaluator.normalize_data(datas, 
                    StockDailySvmModelEvaluator.pct_chg_idx,
                    mus, stds
        )
        StockDailySvmModelEvaluator.normalize_data(datas, 
                    StockDailySvmModelEvaluator.vol_idx,
                    mus, stds
        )
        StockDailySvmModelEvaluator.normalize_data(datas, 
                    StockDailySvmModelEvaluator.amount_idx,
                    mus, stds
        )