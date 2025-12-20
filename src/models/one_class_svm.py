from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

final_df = pd.read_csv(r'C:\Users\matve\OneDrive\Документы\projects\anomaly\data\processed\features.csv')

final_df['block_timestamp'] = pd.to_datetime(final_df['timeStamp'])

first_transaction = final_df.groupby('from')['block_timestamp'].min().reset_index()
first_transaction.rename(columns={'block_timestamp': 'first_seen_timestamp'}, inplace=True)

reference_time = final_df['block_timestamp'].max()
first_transaction['wallet_age_days'] = (reference_time - first_transaction['first_seen_timestamp']).dt.total_seconds() / (24 * 3600)

final_df = final_df.merge(
    first_transaction[['from', 'wallet_age_days']], 
    on='from', 
    how='left'
)
final_df['wallet_age_days'].fillna(0, inplace=True)

final_df_sorted = final_df.sort_values(['from', 'block_timestamp']).copy()

def calculate_7day_activity(group):
    result = []
    for idx, row in group.iterrows():
        current_time = row['block_timestamp']
        time_threshold = current_time - timedelta(days=7)
        
        mask = (group['block_timestamp'] >= time_threshold) & (group['block_timestamp'] < current_time)
        count = mask.sum()
        result.append(count)
    
    return result

tqdm.pandas(desc="Processing addresses")
final_df_sorted['tx_count_7d'] = final_df_sorted.groupby('from').progress_apply(
    lambda x: calculate_7day_activity(x)
).explode().values

final_df = final_df_sorted

final_df['gas_used_ratio'] = final_df['gasUsed'] / final_df['gas']
final_df['gas_used_ratio'] = final_df['gas_used_ratio'].replace([np.inf, -np.inf], 1.0).fillna(1.0)

median_gas_price = final_df['gasPrice'].median()
final_df['gas_price_deviation'] = final_df['gasPrice'] / median_gas_price

final_df['input_length'] = final_df['input'].apply(
    lambda x: len(str(x)[2:]) // 2 if str(x).startswith('0x') else 0
)

final_df['is_new_wallet'] = (final_df['wallet_age_days'] < 7).astype(int)
final_df['is_high_frequency'] = (final_df['tx_count_7d'] > 50).astype(int)

features = [
    'gas_used_ratio', 
    'gas_price_deviation', 
    'input_length',
    'wallet_age_days', 
    'tx_count_7d',
    'is_new_wallet',
    'is_high_frequency'
]

X = final_df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = OneClassSVM(
    kernel='rbf',           
    gamma='scale',         
    nu=0.05,               
    verbose=True            
)

model.fit(X_scaled)

predictions = model.predict(X_scaled)          
anomaly_scores = -model.decision_function(X_scaled) 

final_df['anomaly_score'] = anomaly_scores
final_df['is_anomaly'] = np.where(predictions == -1, 1, 0) 

anomalies = final_df[final_df['is_anomaly'] == 1]
normal = final_df[final_df['is_anomaly'] == 0]

sorted_final_df = final_df.sort_values('anomaly_score', ascending=False)

columns = ['blockHash', 'transactionIndex', 'timeStamp', 'nonce', 'from', 'to', 'eth_value', 'eth_gas_cost', 'gas_used_ratio', 'gas_price_deviation', 'anomaly_score', 'is_anomaly']

available_cols = [col for col in columns if col in sorted_final_df.columns]
filtered_df = sorted_final_df[available_cols]

output_path = r'C:\Users\matve\OneDrive\Документы\projects\anomaly\data\models_data\one_class_svm.csv'
filtered_df.to_csv(output_path, index=False)

print(f"The One-Class SVM model is applied")
print(f"Data saved in {output_path}")
print(f"Total transactions: {len(final_df)}")
print(f"Anomalies found: {len(anomalies)}")
print(f"Normal transactions found: {len(normal)}")
print(f"Anomaly rate: {len(anomalies)/len(final_df)*100:.2f}%")