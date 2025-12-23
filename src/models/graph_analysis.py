import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(r'C:\Users\matve\OneDrive\Документы\projects\anomaly\data\processed\features.csv')


G = nx.DiGraph()

for _, row in df.iterrows():
    G.add_edge(
        row['from'],          
        row['to'],            
        value_eth=row['eth_value'],       
        gas_cost=row['eth_gas_cost'],     
        timestamp=row['timeStamp']        
    )


graph_metrics = {}

for node in G.nodes():
    graph_metrics[node] = {
        'in_degree': G.in_degree(node),
        'out_degree': G.out_degree(node),
        'total_degree': G.in_degree(node) + G.out_degree(node)
    }

pagerank = nx.pagerank(G, alpha=0.85)  
for node in graph_metrics:
    graph_metrics[node]['pagerank'] = pagerank.get(node, 0)

betweenness = nx.betweenness_centrality(G, normalized=True)
for node in graph_metrics:
    graph_metrics[node]['betweenness'] = betweenness.get(node, 0)

G_undirected = G.to_undirected()
clustering = nx.clustering(G_undirected)

for node in graph_metrics:
    graph_metrics[node]['clustering'] = clustering.get(node, 0)

for node in graph_metrics:
    neighbors = set(G.neighbors(node)) | set(G.predecessors(node))
    second_order = set()
    for neighbor in neighbors:
        second_order.update(set(G.neighbors(neighbor)) | set(G.predecessors(neighbor)))
    second_order.discard(node)
    second_order = second_order - neighbors
    graph_metrics[node]['second_order_neighbors'] = len(second_order)

graph_df = pd.DataFrame.from_dict(graph_metrics, orient='index')
graph_df.index.name = 'address'
graph_df.reset_index(inplace=True)

graph_df.to_csv('graph_metrics.csv', index=False)

df_hybrid = df.copy()

df_hybrid = df_hybrid.merge(
    graph_df,
    left_on='from',
    right_on='address',
    how='left',
    suffixes=('', '_from')
)

rename_from = {col: f'{col}_from' for col in graph_df.columns if col != 'address'}
df_hybrid = df_hybrid.rename(columns=rename_from)

df_hybrid = df_hybrid.merge(
    graph_df,
    left_on='to',
    right_on='address',
    how='left',
    suffixes=('', '_to')
)

rename_to = {col: f'{col}_to' for col in graph_df.columns if col != 'address'}
df_hybrid = df_hybrid.rename(columns=rename_to)

if 'address' in df_hybrid.columns:
    df_hybrid = df_hybrid.drop(columns=['address'])
if 'address_from' in df_hybrid.columns:
    df_hybrid = df_hybrid.drop(columns=['address_from'])
if 'address_to' in df_hybrid.columns:
    df_hybrid = df_hybrid.drop(columns=['address_to'])

df_hybrid.to_csv('hybrid_features.csv', index=False)


try:
    top_nodes = sorted(graph_df.nlargest(30, 'total_degree')['address'].tolist())
    subgraph = G.subgraph(top_nodes)
    
    plt.figure(figsize=(12, 10))
    
    node_sizes = [graph_metrics[node]['pagerank'] * 10000 + 100 for node in subgraph.nodes()]
    
    node_colors = [graph_metrics[node]['betweenness'] * 10 for node in subgraph.nodes()]
    
    pos = nx.spring_layout(subgraph, k=1, iterations=50)
    nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                          node_color=node_colors, cmap=plt.cm.Reds, alpha=0.8)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.3, edge_color='gray')
    
    labels = {}
    for node in subgraph.nodes():
        if graph_metrics[node]['pagerank'] > 0.01:  
            labels[node] = node[:6] + '...'  
    nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
    
    plt.title('Транзакционный граф Ethereum\n(Размер: PageRank, Цвет: Betweenness)', fontsize=14)
    plt.axis('off')
    
    plt.text(0.02, 0.02, f"Узлов: {len(subgraph.nodes())}\nРёбер: {len(subgraph.edges())}", 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('transaction_graph.png', dpi=300, bbox_inches='tight')
    plt.savefig('transaction_graph.pdf', bbox_inches='tight')
    plt.show()
    
    print("Граф визуализирован и сохранён в 'transaction_graph.png'")
    
except Exception as e:
    print(f"Визуализация не удалась: {str(e)}")


print("\n СТАТИСТИКА ГРАФОВЫХ МЕТРИК:")

print("\n   ДЛЯ КАЖДОГО КОШЕЛЬКА ВЫЧИСЛЕНЫ:")
for metric in ['in_degree', 'out_degree', 'pagerank', 'betweenness', 'clustering']:
    if metric in graph_df.columns:
        mean_val = graph_df[metric].mean()
        max_val = graph_df[metric].max()
        print(f"   • {metric:25s}: среднее = {mean_val:.6f}, максимум = {max_val:.6f}")

print("\n   ТОП-5 КОШЕЛЬКОВ ПО PAGERANK (важности):")
top_pagerank = graph_df.nlargest(5, 'pagerank')[['address', 'pagerank', 'total_degree']].copy()
top_pagerank['address_short'] = top_pagerank['address'].apply(lambda x: x[:8] + '...' + x[-6:])
for _, row in top_pagerank.iterrows():
    print(f"   {row['address_short']:25s} PageRank: {row['pagerank']:.6f}, Связей: {row['total_degree']}")

print("\n   ТОП-5 КОШЕЛЬКОВ ПО BETWEENNESS (посредничеству):")
top_betweenness = graph_df.nlargest(5, 'betweenness')[['address', 'betweenness', 'total_degree']].copy()
top_betweenness['address_short'] = top_betweenness['address'].apply(lambda x: x[:8] + '...' + x[-6:])
for _, row in top_betweenness.iterrows():
    print(f"   {row['address_short']:25s} Betweenness: {row['betweenness']:.6f}, Связей: {row['total_degree']}")