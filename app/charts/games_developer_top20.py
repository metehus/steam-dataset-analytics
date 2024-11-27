import matplotlib.pyplot as plt

def create_games_developer_top20(df):

    games_per_developer = df['developer'].value_counts().head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    games_per_developer.sort_values().plot(kind='barh', color='lightgreen', edgecolor='black')
    ax.set_title('Número de Jogos por Desenvolvedora (Top 20)', fontsize=16)
    ax.set_xlabel('Número de Jogos', fontsize=12)
    ax.set_ylabel('Desenvolvedoras', fontsize=12)

    return fig