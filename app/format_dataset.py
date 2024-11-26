import re
import pandas as pd

def extract_reviews_info(reviews_text):
    # Extrai quantidade de reviews e % de positivas da coluna
    # Exemplo de valro: "Mostly Positive,(7,030),- 71% of the 7,030 user reviews for this game are positive."
    match = re.search(r"(\d+(?:,\d+)*)", reviews_text)
    num_reviews = int(match.group(1).replace(",", "")) if match else 0
    match_percent = re.search(r"(\d+)%", reviews_text)
    percent_positive = int(match_percent.group(1)) if match_percent else 0
    return num_reviews, percent_positive

def extract_price(price_text):
  # Extrai coluna de preço, retirando o cifrão e lidando de casos de jogos gratis (valor vira Free to play)
   if (isinstance(price_text, str) and "free" in price_text.lower()):
      return 0.0
   
  #  print(price_text)
   try:
      return float(price_text.replace("$", ""))
   except:
      return None

def format_dataset(df):
  data_df = df.copy()
  # removendo linhas que o review ou preco tem nan
  data_df = df.dropna(subset=['all_reviews', 'original_price'])

  # extracao de reviews
  data_df[['num_reviews', 'percent_positive']] = data_df['all_reviews'].apply(
    lambda x: pd.Series(extract_reviews_info(x))
  )

  # parseando colunas que são listas
  data_df['genre_list'] = data_df['genre'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
  data_df['detail_list'] = data_df['game_details'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
  
  data_df['price'] = data_df['original_price'].apply(extract_price)

  # remove os preços invalidos (nan/None), como demos
  data_df = data_df.dropna(subset=['price', 'num_reviews', 'percent_positive'])

  # transformar porcentagem em vetor de 0.0-1.0
  data_df['percent_positive'] = data_df['percent_positive'] / 100.0

  # exportar df em arquivo xlsx para debug
  # data_df.to_excel('output.xlsx', index=False)

  return data_df