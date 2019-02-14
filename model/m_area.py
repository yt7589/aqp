import model.m_mysql as db

class MArea(object):
    def __init__(self):
        self.name = 'MArea'
        
    @staticmethod
    def get_area_id_by_name(area_name):
        sql = 'select area_id from t_area where area_name=%s'
        params = (area_name)
        rowcount, rows = db.query(sql, params)
        if len(rows) > 0:
            return rows[0][0]
        else:
            return -1
            
    @staticmethod
    def add_area(area_name):
        sql = 'insert into t_area(area_name) values(%s)'
        params = (area_name)
        area_id, affected_rows = db.insert(sql, params)
        return area_id