import datetime
import tushare as ts
import pymysql
from app_registry import appRegistry as ar
from model.m_area import MArea
from model.m_industry import MIndustry
from model.m_stock import MStock

'''
获取沪深两市所有挂牌股票基本信息，并保存到数据库中
'''
class CStock(object):
    def __init__(self):
        self.name = 'CStock'
        
    @staticmethod
    def get_stocks():
        ''' 获取市场上所有挂牌股票基本信息 '''
        ts.set_token(ar.ts_token)
        pro = ts.pro_api()
        data = pro.stock_basic(exchange='', list_status='L', 
                    fields='ts_code,symbol,name,area,industry,list_date')
        rec_nums = data.shape[0]
        for j in range(rec_nums):
            rec = list(data.ix[rec_nums - 1 - j])
            flds = []
            area_id = CStock.process_area(rec[3])
            industry_id = CStock.process_industry(rec[4])
            stock_id = CStock.process_stock(rec[0], rec[1], rec[2], 
                        area_id, industry_id, rec[5])
        print('^_^ End caller={0}'.format(ar.caller))
        
    @staticmethod
    def process_area(area_name):
        ''' 
        处理地区信息：如果地区不存在添加到数据库表中，如果存在
        则求出其area_id
        '''
        area_id = MArea.get_area_id_by_name(area_name)
        if area_id>0:
            print('地区存在：{0}'.format(area_name))
        else:
            area_id = MArea.add_area(area_name)
            print('添加地区：{0}---{1}'.format(area_id, area_name))
        return area_id
        
    @staticmethod
    def process_industry(industry_name):
        ''' 
        处理行业信息：如果行业不存在则添加表数据库表中，如果
        存在则返回industry_id
        '''
        industry_id = MIndustry.get_industry_id_by_name(industry_name)
        if industry_id>0:
            print('行业存在：{0}'.format(industry_name))
        else:
            industry_id = MIndustry.add_industry(industry_name)
            print('添加行业：{0}---{1}'.format(industry_id, industry_name))
        return industry_id
        
    @staticmethod
    def process_stock(ts_code, symbol, stock_name, area_id, 
                industry_id, list_date):
        '''
        处理股票基本信息：如果股票不存在则添加到数据库中，如果存在
        则返回股票编号
        '''
        stock_id = MStock.get_stock_id_by_name(stock_name)
        if stock_id > 0:
            print('股票存在：{0}'.format(stock_name))
        else:
            stock_id = MStock.add_stock(ts_code, symbol, 
                        stock_name, area_id, industry_id, list_date)
            print('添加股票：{0}---{1}'.format(stock_id, stock_name))
        return stock_id
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        