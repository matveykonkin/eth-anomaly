import requests
import pandas as pd

ip_key = 'PC56519M1IDXI9AYEQD1IB98BD4PBT8N43'
addresses = [
    "0xa6bccfd21c5b23965c5f31d70fcef17b83b9c04a", # 1
    "0xe2102860361ced07ab9d75dbe815962b72165e82", # 2
    "0x186e6139cd55902faa3d70b4cd866e7237d2f6ed", # 3
    "0x5ced8dc729662eab1f923b3c43bfe6bb2dcd2ae6", # 4 
    "0x74d1cdAB3D434C610beFa65C3bB30F602846939e", # 5
    "0x83Eb78d526E4B5e4623D9123F32f1c118Fd1722C", # 6
    "0x5a52E96BAcdaBb82fd05763E25335261B270Efcb", # 7
]

# 1-4 addresses blacklisted by Tether, 5-6 wallets with suspicious activity, 7 Amber Group wallet

all_data = []

for address in addresses:
    url = 'https://api.etherscan.io/v2/api'

    params = {
        'module': 'account',
        'action': 'txlist',
        'address': address,
        'apikey': ip_key,
        'chainid': 1,
        'offset': 10000,
        'page': 1,
        'sort': 'desc'
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get('status') == '1' and data.get('result'):
        transactions = data['result']
        print(f"Finded transactions: {len(transactions)}")
        
        for tx in transactions:
            tx['wallet_address'] = address
            
        all_data.extend(transactions)
    else:
        print(f"No data for address: {address}")
        print(f"Answer API: {data}")

if all_data:
    df = pd.DataFrame(all_data)
    df.to_csv('data/raw/etherscan_data.csv', index=False)
    print(f"All data saved! Size: {df.shape}")
    print(f"Unique address: {df['wallet_address'].nunique()}")
else:
    print("No data to save")