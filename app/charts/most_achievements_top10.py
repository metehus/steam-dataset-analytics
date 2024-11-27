import pandas as pd
import matplotlib.pyplot as plt

def create_most_achievements_chart_top10(df):
    
    df['achievements'] = pd.to_numeric(df['achievements'], errors='coerce').fillna(0).astype(int)
    
    top_10_achievements = df.nlargest(10, 'achievements')[['name', 'achievements']]
    
    # Criando o gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.barh(
        top_10_achievements.iloc[::-1]['name'],
        top_10_achievements.iloc[::-1]['achievements'],
        color='skyblue', 
        edgecolor='black' 
    )
    ax.set_title('Top 10 Jogos com Mais Conquistas', fontsize=16)
    ax.set_xlabel('Número de Conquistas', fontsize=12)
    ax.set_ylabel('Jogos', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)

    return fig
