import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import numpy as np

df_original = pd.read_csv('data/processed/features.csv') 
df_hybrid = pd.read_csv('data/graph_data/hybrid_features.csv')  

original_features = ['eth_value', 'eth_gas_cost', 'gas', 'gasPrice', 'gasUsed']
X_original = df_original[original_features].fillna(0)

numeric_cols = df_hybrid.select_dtypes(include=[np.number]).columns
X_hybrid = df_hybrid[numeric_cols].fillna(0)


if_original = IsolationForest(contamination=0.1, random_state=42)
if_original_pred = if_original.fit_predict(X_original)

svm_original = OneClassSVM(nu=0.1)
svm_original_pred = svm_original.fit_predict(X_original)

if_hybrid = IsolationForest(contamination=0.1, random_state=42)
if_hybrid_pred = if_hybrid.fit_predict(X_hybrid)

svm_hybrid = OneClassSVM(nu=0.1)
svm_hybrid_pred = svm_hybrid.fit_predict(X_hybrid)

results = pd.DataFrame({
    'Модель': ['IF (без графов)', 'SVM (без графов)', 'IF (+графы)', 'SVM (+графы)'],
    'Аномалий': [
        (if_original_pred == -1).sum(),
        (svm_original_pred == -1).sum(),
        (if_hybrid_pred == -1).sum(),
        (svm_hybrid_pred == -1).sum()
    ],
    'Процент': [
        f"{(if_original_pred == -1).mean()*100:.1f}%",
        f"{(svm_original_pred == -1).mean()*100:.1f}%",
        f"{(if_hybrid_pred == -1).mean()*100:.1f}%",
        f"{(svm_hybrid_pred == -1).mean()*100:.1f}%"
    ]
})

print("\nРЕЗУЛЬТАТЫ:")
print(results)

df_hybrid['if_anomaly_hybrid'] = np.where(if_hybrid_pred == -1, 1, 0)
df_hybrid['svm_anomaly_hybrid'] = np.where(svm_hybrid_pred == -1, 1, 0)
df_hybrid.to_csv('hybrid_with_predictions.csv', index=False)