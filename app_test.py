
def main():
    print('test main...')
    etf_names = ['ICBC', 'CBC']
    # 
    etfs = []
    for etf_name in etf_names:
        etf = {'name': etf_name, 'volume': 0, 'amount': 0}
        etfs.append(etf)
        
    for ei in etfs:
        print('{0}; {1}; {2}'.format(ei['name'], ei['volume'], ei['amount']))
       
    
if '__main__' == __name__:
    main()