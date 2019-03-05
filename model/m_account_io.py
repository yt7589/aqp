import model.m_mysql as db

class MAccountIo(object):
    IO_IN = 1
    IO_OUT = 2

    def __init__(self):
        self.name = 'MAccountIo'

    @staticmethod
    def get_latest_io(account_id):
        '''
        找出指定账户最新一条流水记录
        @version v0.0.1 闫涛 2019.03.02
        '''
        sql = 'select balance from t_account_io where account_id=%s '\
                    'and account_io_id = (select max(account_io_id) '\
                    'from t_account_io where account_id=%s)'
        params = (account_id, account_id)
        return db.query(sql, params)

    @staticmethod
    def withdraw(account_id, amount):
        '''
        从指定账户取出一笔钱
        @version v0.0.1 闫涛 2019-03-03
        '''
        return MAccountIo._do_account_io(account_id, MAccountIo.IO_OUT, amount)

    @staticmethod
    def _do_account_io(account_id, io_type, amount):
        rc, rows = MAccountIo.get_latest_io(account_id)
        balance = 0
        if rc <= 0:
            return False
        balance = rows[0][0]
        if MAccountIo.IO_OUT==io_type and amount>balance:
            return False
        sql = 'insert into t_account_io(account_id, io_date, '\
                    'in_amount, out_amount, balance) values(%s, '\
                        'sysdate(), %s, %s, %s)'
        if io_type == MAccountIo.IO_IN:
            params = (account_id, amount, 0.0, balance+amount)
        else:
            params = (account_id, 0.0, amount, balance-amount)
        account_io_id, affected_rows = db.insert(sql, params)
        if 1 == affected_rows:
            return True
        else:
            return False