import matplotlib.pyplot as plt
import re

def create_developer_rating_chart_top5(df):
    df['review_score'] = df['num_reviews']

    developer_ratings = (
    df.groupby('developer')['review_count']
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

    # Criar o gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    developer_ratings.sort_values().plot(kind='barh', color='skyblue', edgecolor='black')
    ax.set_title('Top Desenvolvedoras por Avaliações (Top 5)', fontsize=16)
    ax.set_xlabel('Número de Avaliações(Milhões)', fontsize=12)
    ax.set_ylabel('Desenvolvedoras', fontsize=12)
    # ax.set_grid(axis='x', linestyle='--', alpha=0.7)
    # ax.set_tight_layout()
 
    return fig
