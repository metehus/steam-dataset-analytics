import json
from django.shortcuts import render
from django.http import HttpResponse, FileResponse, Http404, JsonResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
from app.format_dataset import format_dataset

from app.charts.genre_popularity_top5 import create_genre_popularity_chart_top5
from app.charts.genre_popularity_top10 import create_genre_popularity_chart_top10
from app.charts.genre_popularity_top20 import create_genre_popularity_chart_top20

from app.charts.top_developers_reviews_top5 import create_developer_rating_chart_top5
from app.charts.top_developers_reviews_top10 import create_developer_rating_chart_top10
from app.charts.top_developers_reviews_top20 import create_developer_rating_chart_top20

from app.charts.games_reviews_top5 import create_games_reviews_top5
from app.charts.games_reviews_top10 import create_games_reviews_top10
from app.charts.games_reviews_top20 import create_games_reviews_top20

from app.charts.games_developer_top5 import create_games_developer_top5
from app.charts.games_developer_top10 import create_games_developer_top10
from app.charts.games_developer_top20 import create_games_developer_top20

from app.charts.genre_distribution_developer_top5 import create_genre_distribution_developer_top5
from app.charts.genre_distribution_developer_top10 import create_genre_distribution_developer_top10
from app.charts.genre_distribution_developer_top20 import create_genre_distribution_developer_top20

from app.train import is_trained, predict_positive_review, train

matplotlib.use('Agg')

media_dir = "media"
data_dir = "data"

df_file_path = os.path.join(data_dir, 'df_data.pkl')
model_path = os.path.join(data_dir, 'model.joblib')
# Create your views here.

def index_view(request):
  template = loader.get_template('index.html')
  return HttpResponse(template.render())



@csrf_exempt
def generate_charts(request):
    try:
      os.makedirs(data_dir, exist_ok=True)
      os.makedirs(media_dir, exist_ok=True)
      uploaded_file = request.FILES['file']

      df = pd.read_csv(uploaded_file)
      df = format_dataset(df)
      df.to_pickle(df_file_path)

      # cria os plots e salva eles nesse objeto
      charts = {
          'genre_popularity_top5': create_genre_popularity_chart_top5(df),
          'genre_popularity_top10': create_genre_popularity_chart_top10(df),
          'genre_popularity_top20': create_genre_popularity_chart_top20(df),
          'developer_rating_chart_top5' : create_developer_rating_chart_top5(df),
          'developer_rating_chart_top10' : create_developer_rating_chart_top10(df),
          'developer_rating_chart_top20' : create_developer_rating_chart_top20(df),
          'genre_distribution_developer_top5' : create_genre_distribution_developer_top5(df),
          'genre_distribution_developer_top10' : create_genre_distribution_developer_top10(df),
          'genre_distribution_developer_top20' : create_genre_distribution_developer_top20(df),
          'games_developer_top5' : create_games_developer_top5(df),
          'games_developer_top10' : create_games_developer_top10(df),
          'games_developer_top20' : create_games_developer_top20(df),
          'games_reviews_top5' : create_games_reviews_top5(df),
          'games_reviews_top10' : create_games_reviews_top10(df),
          'games_reviews_top20' : create_games_reviews_top20(df)
      }

      response = {}
      
      # exporta cada plot na pasta
      for name, figure in charts.items():
        filename = f'{name}.jpg'
        output_file = os.path.join(media_dir, filename)
        figure.savefig(output_file, format='jpg', dpi=300, bbox_inches='tight')
        plt.close(figure)

        response[name] = filename


      return JsonResponse(response)
    except Exception as e:
      print(e)
      return JsonResponse({
        'erro': str(e)
      }, status=500)

@csrf_exempt
def get_train_values(request):
  # endpoint que retorna se ja foi treinado, e valores disponiveis para predição
  df = pd.read_pickle(df_file_path)
  
  if df is None:
    raise 'No df selected'
     
  unique_genres = (
    df['genre']
    .dropna()
    .str.split(',')
    .explode()
    .str.strip()
    .unique()     
  )

  unique_details = (
    df['game_details']
    .dropna()
    .str.split(',')
    .explode()
    .str.strip()
    .unique()     
  )

  response = {
    'trained': is_trained(),
    'genres': unique_genres.tolist(),
    'details': unique_details.tolist(),
  }

  return JsonResponse(response)
    
@csrf_exempt
def train_view(request):
  # endpoint de treino
  df = pd.read_pickle(df_file_path)
  
  if df is None:
    raise 'No df selected'
  
  train_metrics = train(df)

  return JsonResponse(train_metrics)

@csrf_exempt
def predict_view(request):
  # endpoint de predição
  if request.method != 'POST':
    return JsonResponse({'error': 'Invalid HTTP method'}, status=405)
  try:
    data = json.loads(request.body)
    
    price = data.get('price', 0)
    detail = data.get('detail', '')
    genre = data.get('genre', '')

    predicted_percent = predict_positive_review(price, detail, genre)

    return JsonResponse({'percent': predicted_percent})

  except Exception as e:
    return JsonResponse({'error': str(e)}, status=400)

   

def serve_media_file(request, filename):
    # endpoint para servir imagens dos graficos
    file_path = os.path.join(media_dir, filename)

    if not os.path.exists(file_path):
        raise Http404("File not found")

    response = FileResponse(open(file_path, 'rb'))
    return response