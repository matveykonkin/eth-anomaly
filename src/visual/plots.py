import pandas as pd
import matplotlib.pyplot as plt

# ====== 1. Isolation Forest ======
df_if = pd.read_csv(r'C:\Users\matve\OneDrive\Документы\projects\anomaly\data\models_data\iso_forest.csv')

anomalies_if = df_if[df_if['anomaly_score'] < 0]
normal_if = df_if[df_if['anomaly_score'] >= 0]

plt.figure(figsize=(12, 6))
plt.hist(normal_if['anomaly_score'], bins=50, alpha=0.7, label='Нормальные', 
         color='blue', density=True)
plt.hist(anomalies_if['anomaly_score'], bins=50, alpha=0.7, label='Аномалии', 
         color='red', density=True)
plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Граница')
plt.xlabel('Anomaly Score')
plt.ylabel('Плотность')
plt.title('Isolation Forest')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('iso_forest.png')
plt.show()

# ====== 2. One-Class SVM ======
df_svm = pd.read_csv(r'C:\Users\matve\OneDrive\Документы\projects\anomaly\data\models_data\one_class_svm.csv')

anomalies_svm = df_svm[df_svm['is_anomaly'] == 1]
normal_svm = df_svm[df_svm['is_anomaly'] == 0]

plt.figure(figsize=(12, 6))
plt.hist(normal_svm['anomaly_score'], bins=50, alpha=0.7, label='Нормальные', 
         color='blue', density=True)
plt.hist(anomalies_svm['anomaly_score'], bins=50, alpha=0.7, label='Аномалии', 
         color='red', density=True)
plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Граница')
plt.xlabel('Anomaly Score')
plt.ylabel('Плотность')
plt.title('One-Class SVM')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('one_class_svm.png')
plt.show()