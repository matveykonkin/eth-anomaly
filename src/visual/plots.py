import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Isolation Forest model
labeled_df = pd.read_csv(r'C:\Users\matve\OneDrive\Документы\projects\anomaly\data\models_data\iso_forest.csv')

anomalies = labeled_df[labeled_df['is_anomaly'] == -1]
normal = labeled_df[labeled_df['is_anomaly'] == 1]

plt.figure(figsize=(12, 6))

plt.hist(normal['anomaly_score'], bins=50, alpha=0.7, label='Нормальные', 
         color='blue', density=True)
plt.hist(anomalies['anomaly_score'], bins=50, alpha=0.7, label='Аномалии', 
         color='red', density=True)

plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Граница аномалий')
plt.xlabel('Anomaly Score')
plt.ylabel('Плотность распределения')
plt.title('Распределение оценок аномальности')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('iso_forest.png', dpi=300, bbox_inches='tight')
plt.show()