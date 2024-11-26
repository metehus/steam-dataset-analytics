import matplotlib.pyplot as plt
import re


def create_genre_popularity_chart(df):
  df['review_count'] = df['num_reviews']
    
  genre_counts = (
    df['genre']
      .dropna()
      .str.split(',')
      .explode()
      .value_counts()
      .head(10)
  )

  fig, ax = plt.subplots(figsize=(10, 6))
  genre_counts.plot.pie(
      autopct='%1.1f%%',
      startangle=140,
      colors=plt.cm.tab10.colors,
      textprops={'fontsize': 10},
      ax=ax
  )
  ax.set_title('Popularidade por Gênero (Top 10)', fontsize=16)
  ax.set_ylabel('')  # Remove y-label for pie chart aesthetics

  return fig