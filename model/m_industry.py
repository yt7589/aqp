import model.m_mysql as db

class MIndustry(object):
    def __init__(self):
        self.name = 'MIndustry'
        
    @staticmethod
    def get_industry_id_by_name(industry_name):
        sql = 'select industry_id from t_industry where industry_name=%s'
        params = (industry_name)
        rowcount, rows = db.query(sql, params)
        if len(rows) > 0:
            return rows[0][0]
        else:
            return -1
            
    @staticmethod
    def add_industry(industry_name):
        sql = 'insert into t_industry(industry_name) values(%s)'
        params = (industry_name)
        industry_id, affected_rows = db.insert(sql, params)
        return industry_id