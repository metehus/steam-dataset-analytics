import pandas as pd
import matplotlib.pyplot as plt

def create_most_achievements_chart_top20(df):
    
    df['achievements'] = pd.to_numeric(df['achievements'], errors='coerce').fillna(0).astype(int)
    
    top_5_achievements = (
        df[['name', 'achievements']]
        .sort_values(by='achievements', ascending=False)
        .head(20)
    )
    
    # Plotar o gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        top_5_achievements['name'],
        top_5_achievements['achievements'],
        color=plt.cm.viridis.colors[:5]
    )
    ax.set_title('Top 5 Jogos com Mais Conquistas', fontsize=16)
    ax.set_xlabel('Jogos', fontsize=12)
    ax.set_ylabel('Número de Conquistas', fontsize=12)
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return fig
