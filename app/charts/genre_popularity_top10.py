import matplotlib.pyplot as plt
import re


def create_genre_popularity_chart_top10(df):
    df['review_count'] = df['num_reviews']
    
    # Conta as ocorrências de cada gênero e seleciona os 20 principais
    genre_counts = (
        df['genre']
          .dropna()
          .str.split(',')
          .explode()
          .value_counts()
          .head(10)
    )

    # Cria o gráfico de barras horizontal
    fig, ax = plt.subplots(figsize=(12, 8))
    genre_counts.plot.barh(
        color=plt.cm.tab10.colors,  # Paleta de cores
        ax=ax
    )
    
    # Configuração do gráfico
    ax.set_title('Popularidade por Gênero (Top 10)', fontsize=16)
    ax.set_xlabel('Quantidade', fontsize=12)
    ax.set_ylabel('Gêneros', fontsize=12)
    ax.invert_yaxis()  # Inverte a ordem para exibir o maior no topo

    return fig
