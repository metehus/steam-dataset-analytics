import matplotlib.pyplot as plt

def create_genre_distribution_developer_top20(df):
    # Processando dados de gênero e desenvolvedor
    genre_data = (
        df[['developer', 'genre']]
        .dropna()
        .assign(genre=lambda x: x['genre'].str.split(','))  # Divide os gêneros por vírgula
        .explode('genre')  # Expande os gêneros em linhas separadas
    )
    
    # Contando o número de jogos por desenvolvedor
    developer_game_counts = genre_data.groupby('developer').size()
    top_developers = developer_game_counts.nlargest(20).index  # Top 10 desenvolvedores com mais jogos

    # Contando a distribuição de gêneros
    genre_counts = genre_data.groupby(['developer', 'genre']).size().unstack(fill_value=0)
    genre_counts = genre_counts.loc[top_developers]  # Filtrando para os top desenvolvedores

    # Normalizando os dados para proporção
    genre_counts_normalized = genre_counts.div(genre_counts.sum(axis=1), axis=0)

    # Criar o gráfico
    fig, ax = plt.subplots(figsize=(12, 8))
    genre_counts_normalized.plot(kind='bar', stacked=True, colormap='tab20', ax=ax)

    # Configurações do gráfico
    ax.set_title('Distribuição de Gêneros por Desenvolvedor (Top 20)', fontsize=16)
    ax.set_xlabel('Desenvolvedores', fontsize=12)
    ax.set_ylabel('Proporção de Gêneros', fontsize=12)
    ax.legend(title='Gêneros', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    return fig
