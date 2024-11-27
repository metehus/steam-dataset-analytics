import matplotlib.pyplot as plt

def create_games_developer_top5(df):

    games_per_developer = df['developer'].value_counts().head(5)

    fig, ax = plt.subplots(figsize=(10, 6))
    games_per_developer.sort_values().plot(kind='barh', color='lightgreen', edgecolor='black')
    ax.set_title('Número de Jogos por Desenvolvedora (Top 5)', fontsize=16)
    ax.set_xlabel('Número de Jogos', fontsize=12)
    ax.set_ylabel('Desenvolvedoras', fontsize=12)

    return fig