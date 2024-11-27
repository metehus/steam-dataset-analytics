import matplotlib.pyplot as plt

def create_games_reviews_top5(df):

    # Filtrando os 10 jogos mais bem avaliados
    top_games = df.nlargest(5, 'num_reviews')[['name', 'num_reviews']]

    # Criando o gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.barh(top_games.iloc[::-1]['name'], top_games.iloc[::-1]['num_reviews'], color='orange', edgecolor='black')
    ax.set_title('Top Jogos por Avaliações (Top 5)', fontsize=16)
    ax.set_xlabel('Número de Avaliações', fontsize=12)
    ax.set_ylabel('Jogos', fontsize=12)

    return fig