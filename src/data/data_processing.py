import pandas as pd
import numpy as np
import datetime
import os

def load_raw_data():
    raw_data = pd.read_csv('data/raw/etherscan_data.csv')
    
    return raw_data

def clean_data(raw_data):
    clean_data = raw_data.copy()
    clean_data['timeStamp'] = pd.to_datetime(clean_data['timeStamp'], unit='s')    
    numbers_data = ['value', 'gas', 'gasPrice', 'gasUsed', 'nonce', 'blockNumber']
    
    for col in numbers_data:
        clean_data[col] = clean_data[col].astype('float64')
    
    clean_data = clean_data.drop_duplicates(subset=['hash'])
    clean_data = clean_data[clean_data['isError'] == 0]
    
    return clean_data

def translation_data(clean_data):
    clean_data['eth_value'] = clean_data['value'] / 10**18
    clean_data['eth_gas_cost'] = (clean_data['gasUsed'] * clean_data['gasPrice']) / 10**18
    clean_data['day_of_week'] = clean_data['timeStamp'].dt.day_name()
    clean_data['hour'] = clean_data['timeStamp'].dt.hour

    return clean_data

def save_data(data_frame):
    data_frame.to_csv('data/processed/features.csv', index=False)

if __name__ == "__main__":
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    final_df = translation_data(cleaned_df)
    save_data(final_df)
    print('processing completed.')