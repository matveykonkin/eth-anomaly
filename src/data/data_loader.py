import requests
import pandas as pd

url = 'https://api.etherscan.io/v2/api'
ip_key = 'PC56519M1IDXI9AYEQD1IB98BD4PBT8N43'
address = '0x28C6c06298d514Db089934071355E5743bf21d60'

params = {
    'module': 'account',
    'action': 'txlist',
    'address': address,
    'apikey': ip_key,
    'chainid': 1,
    'offset': 3000,
    'page': 1,
    'sort': 'asc'
}

response = requests.get(url, params=params)

data = response.json()

if data.get('status') == '1' and data.get('result'):
    
    raw_df = pd.DataFrame(data['result'])
    raw_df.to_csv('data/raw/etherscan_data.csv', index=False)
    
    print('raw data is saved in data/raw/etherscan_raw.csv')
    print(f'data size: {raw_df.shape}')
    
else:
    print("no data to save")