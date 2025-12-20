import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_if = pd.read_csv(r'C:\Users\matve\OneDrive\Документы\projects\anomaly\data\models_data\iso_forest.csv')
df_svm = pd.read_csv(r'C:\Users\matve\OneDrive\Документы\projects\anomaly\data\models_data\one_class_svm.csv')

df_if['tx_key'] = df_if['blockHash'] + '_' + df_if['transactionIndex'].astype(str)
df_svm['tx_key'] = df_svm['blockHash'] + '_' + df_svm['transactionIndex'].astype(str)

merged = pd.merge(df_if[['tx_key', 'is_anomaly']], 
                  df_svm[['tx_key', 'is_anomaly']], 
                  on='tx_key', 
                  suffixes=('_if', '_svm'))

total_if = len(df_if)
total_svm = len(df_svm)
total_merged = len(merged)

if_anomalies_total = (df_if['is_anomaly'] == -1).sum()
svm_anomalies_total = (df_svm['is_anomaly'] == 1).sum()

# Консольный вывод на английском
print("=== RESULTS ===")
print(f"Total IF transactions: {total_if}")
print(f"Total SVM transactions: {total_svm}")
print(f"Successfully merged: {total_merged}")
print(f"IF found anomalies: {if_anomalies_total}")
print(f"SVM found anomalies: {svm_anomalies_total}")

both_anomaly = ((merged['is_anomaly_if'] == -1) & (merged['is_anomaly_svm'] == 1)).sum()
both_normal = ((merged['is_anomaly_if'] != -1) & (merged['is_anomaly_svm'] != 1)).sum()
if_only = ((merged['is_anomaly_if'] == -1) & (merged['is_anomaly_svm'] != 1)).sum()
svm_only = ((merged['is_anomaly_if'] != -1) & (merged['is_anomaly_svm'] == 1)).sum()

print("\n=== MATCHES ===")
print(f"Both: ANOMALY: {both_anomaly}")
print(f"Both: NORMAL: {both_normal}")
print(f"Only IF: {if_only}")
print(f"Only SVM: {svm_only}")

print("\n=== VERIFICATION ===")
print(f"IF anomalies in merged data: {both_anomaly + if_only} (should be ≤ {if_anomalies_total})")
print(f"SVM anomalies in merged data: {both_anomaly + svm_only} (should be ≤ {svm_anomalies_total})")

agreement = (both_anomaly + both_normal) / total_merged * 100 if total_merged > 0 else 0
svm_coverage = both_anomaly / svm_anomalies_total * 100 if svm_anomalies_total > 0 else 0
if_coverage = both_anomaly / if_anomalies_total * 100 if if_anomalies_total > 0 else 0

print("\n=== SUMMARY ===")
print(f"Total model agreement: {agreement:.1f}%")
print(f"From {svm_anomalies_total} SVM anomalies, IF also found: {both_anomaly} ({svm_coverage:.1f}%)")
print(f"From {if_anomalies_total} IF anomalies, SVM also found: {both_anomaly} ({if_coverage:.1f}%)")

# Графики на русском языке
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
data = [[both_normal, if_only],
        [svm_only, both_anomaly]]

sns.heatmap(data, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=['Норма', 'Аномалия'],
            yticklabels=['Норма', 'Аномалия'],
            cbar_kws={'label': 'Количество транзакций'})
plt.title('Матрица совпадений моделей\nСтроки: IF, Столбцы: SVM')
plt.xlabel('One-Class SVM')
plt.ylabel('Isolation Forest')

plt.subplot(1, 3, 2)
categories = ['Обе модели', 'Только IF', 'Только SVM']
values = [both_anomaly, if_only, svm_only]
colors = ['red', 'blue', 'orange']

bars = plt.bar(categories, values, color=colors, edgecolor='black')
plt.title('Распределение обнаруженных аномалий')
plt.ylabel('Количество')
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(value), ha='center', fontweight='bold')

plt.subplot(1, 3, 3)
labels = ['Совпадают', 'Не совпадают']
sizes = [both_anomaly + both_normal, if_only + svm_only]
colors = ['lightgreen', 'lightcoral']
explode = (0.1, 0)

plt.pie(sizes, labels=labels, colors=colors, explode=explode,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.title(f'Общее согласие: {agreement:.1f}%')

plt.tight_layout()
plt.savefig('comparison_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

if 'anomaly_score' in df_if.columns and 'anomaly_score' in df_svm.columns:
    merged_scores = pd.merge(
        df_if[['tx_key', 'anomaly_score']],
        df_svm[['tx_key', 'anomaly_score']],
        on='tx_key',
        suffixes=('_if', '_svm')
    )
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(df_if['anomaly_score'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Граница (0)')
    axes[0].set_xlabel('Anomaly Score')
    axes[0].set_ylabel('Количество')
    axes[0].set_title('Isolation Forest\nРаспределение оценок')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(df_svm['anomaly_score'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Граница (0)')
    axes[1].set_xlabel('Anomaly Score')
    axes[1].set_ylabel('Количество')
    axes[1].set_title('One-Class SVM\nРаспределение оценок')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    scatter = axes[2].scatter(merged_scores['anomaly_score_if'], 
                            merged_scores['anomaly_score_svm'],
                            alpha=0.6, s=10)
    axes[2].set_xlabel('IF Score')
    axes[2].set_ylabel('SVM Score')
    axes[2].set_title('Сравнение оценок двух моделей')
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scores_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

if 'timeStamp' in df_if.columns and 'timeStamp' in df_svm.columns:
    df_if['date'] = pd.to_datetime(df_if['timeStamp']).dt.date
    df_svm['date'] = pd.to_datetime(df_svm['timeStamp']).dt.date
    
    if_daily = df_if.groupby('date')['is_anomaly'].apply(lambda x: (x == -1).mean() * 100)
    svm_daily = df_svm.groupby('date')['is_anomaly'].apply(lambda x: (x == 1).mean() * 100)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(if_daily.index, if_daily.values, marker='o', label='Isolation Forest', linewidth=2)
    plt.plot(svm_daily.index, svm_daily.values, marker='s', label='One-Class SVM', linewidth=2)
    
    plt.xlabel('Дата')
    plt.ylabel('% аномалий')
    plt.title('Динамика обнаружения аномалий по дням')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('anomalies_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()

fig, ax = plt.subplots(figsize=(10, 6))

metrics = ['Общее\nсогласие', 'Согласие по\nаномалиям SVM', 'Согласие по\nаномалиям IF']
values = [agreement, svm_coverage, if_coverage]
colors = ['skyblue', 'lightcoral', 'lightgreen']

bars = ax.bar(metrics, values, color=colors, edgecolor='black')
ax.set_ylabel('Процент (%)')
ax.set_title('Сравнение метрик согласия моделей')
ax.set_ylim(0, 100)

for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('src/visual/metrics_comparison_isoF_ocsvm.png', dpi=300, bbox_inches='tight')
plt.show()