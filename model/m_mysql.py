#import pymysql
from DBUtils.PooledDB import PooledDB

RD_DBP = PooledDB(
        creator=None, #pymysql,  # 使用链接数据库的模块
        maxconnections=6,  # 连接池允许的最大连接数，0和None表示不限制连接数
        mincached=2,  # 初始化时，链接池中至少创建的空闲的链接，0表示不创建
        maxcached=5,  # 链接池中最多闲置的链接，0和None不限制
        maxshared=3,   #这个参数没多大用，  最大可以被大家共享的链接
        # 链接池中最多共享的链接数量，0和None表示全部共享。PS: 无用，因为pymysql和MySQLdb等模块的 threadsafety都为1，所有值无论设置为多少，_maxcached永远为0，所以永远是所有链接都共享。
        blocking=True,  # 连接池中如果没有可用连接后，是否阻塞等待。True，等待；False，不等待然后报错
        maxusage=None,  # 一个链接最多被重复使用的次数，None表示无限制
        setsession=[],  # 开始会话前执行的命令列表。如：["set datestyle to ...", "set time zone ..."]
        ping=0,
        # ping MySQL服务端，检查是否服务可用。# 如：0 = None = never, 1 = default = whenever it is requested, 2 = when a cursor is created, 4 = when a query is executed, 7 = always
        host='127.0.0.1',
        port=3306,
        user='quant',
        password='Quant2019',
        database='QuantDb',#链接的数据库的名字
        charset='utf8'
    )

WT_DBP = PooledDB(
        creator=None, #pymysql,  # 使用链接数据库的模块
        maxconnections=6,  # 连接池允许的最大连接数，0和None表示不限制连接数
        mincached=2,  # 初始化时，链接池中至少创建的空闲的链接，0表示不创建
        maxcached=5,  # 链接池中最多闲置的链接，0和None不限制
        maxshared=3,   #这个参数没多大用，  最大可以被大家共享的链接
        # 链接池中最多共享的链接数量，0和None表示全部共享。PS: 无用，因为pymysql和MySQLdb等模块的 threadsafety都为1，所有值无论设置为多少，_maxcached永远为0，所以永远是所有链接都共享。
        blocking=True,  # 连接池中如果没有可用连接后，是否阻塞等待。True，等待；False，不等待然后报错
        maxusage=None,  # 一个链接最多被重复使用的次数，None表示无限制
        setsession=[],  # 开始会话前执行的命令列表。如：["set datestyle to ...", "set time zone ..."]
        ping=0,
        # ping MySQL服务端，检查是否服务可用。# 如：0 = None = never, 1 = default = whenever it is requested, 2 = when a cursor is created, 4 = when a query is executed, 7 = always
        host='127.0.0.1',
        port=3306,
        user='quant',
        password='Quant2019',
        database='QuantDb',#链接的数据库的名字
        charset='utf8'
    )

def get_rdb_connection():
    return RD_DBP.connection()

def get_wdb_connection():
    return WT_DBP.connection()

def close_db_connection(conn):
    conn.close()

def query(sql, params):
    conn = get_rdb_connection()
    result = query_t(conn, sql, params)
    close_db_connection(conn)
    return result

def query_t(conn, sql, params):
    cursor = conn.cursor()
    cursor.execute(sql, params)
    rowcount = cursor.rowcount
    rows = cursor.fetchall()
    cursor.close()
    return (rowcount, rows)
    
def insert(sql, params):
    conn = get_wdb_connection()
    result = insert_t(conn, sql, params)
    close_db_connection(conn)
    return result

def insert_t(conn, sql, params):
    cursor = conn.cursor()
    affected_rows = cursor.execute(sql, params)
    conn.commit()
    cursor.close()
    pk = cursor.lastrowid
    return (pk, affected_rows)
    
def delete(sql, params):
    conn = get_wdb_connection()
    result = delete_t(conn, sql, params)
    close_db_connection(conn)
    return result
    
def delete_t(conn, sql, params):
    cursor = conn.cursor()
    affected_rows = cursor.execute(sql, params)
    conn.commit()
    cursor.close()
    return (0, affected_rows)
    
def update(sql, params):
    conn = get_wdb_connection()
    result = update_t(conn, sql, params)
    close_db_connection(conn)
    return result
    
def update_t(conn, sql, params):
    cursor = conn.cursor()
    affected_rows = cursor.execute(sql, params)
    conn.commit()
    cursor.close()
    return (0, affected_rows)
