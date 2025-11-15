import pandas as pd
import matplotlib.pyplot as plt

final_df = pd.read_csv('data/processed/features.csv')

plt.hist(final_df['eth_value'], bins=50)
plt.title('Распределение сумм транзакций')
plt.xlabel('Сумма в ETH')
plt.ylabel('Количество транзакций')

plt.show()

plt.hist(final_df['eth_gas_cost'], bins=50)
plt.title('Распределение стоимости газа')
plt.show()