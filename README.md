# aqp
arxanfintech quant platform

## 股票量化交易研究平台
通常研究股票量化交易是在宽客网站中进行，但是宽客网站仅提供基本的Python编程环境，如numpy、pandas、matplotlib等，这对于进行一般的策略研究是非常方便的，但是如果要进行深度学习的策略研究，由于这些平台不支持TensorFlow或PyTorch，所以不能进行深度学习策略研究。实际上，基于深度学习的量化交易策略研究，还没有成为一种主流的量化交易技术。本平台旨在本地搭建一个完整的股票量化交易研究平台，可以使用TensorFlow等深度学习框架，进行量化交易策略研究，支持完整的回测，用于评估策略的优劣。
## 安装步聚
### 恢复初始状态
在实际开发中，我们经常会需要重置系统为初始状态，可以通过运行下面的SQL语句来实现：
```sql
delete from t_user_stock_io;
delete from t_user_stock;
delete from t_account_io where account_io_id>1;
update t_account set cash_amount=100000000, stock_amount=0 where account_id=1;
```










