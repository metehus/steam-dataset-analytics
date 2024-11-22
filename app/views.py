from django.shortcuts import render
from django.http import HttpResponse, FileResponse, Http404, JsonResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
from app.charts.genre_popularity import create_genre_popularity_chart
from app.charts.top_developers_reviews import create_developer_rating_chart
from app.charts.games_reviews import create_games_reviews
from app.charts.games_developer import create_games_developer
from app.charts.genre_distribution_developer import create_genre_distribution_developer

matplotlib.use('Agg')

media_dir = "media"

df_file_path = os.path.join(media_dir, 'df_data.pkl')
# Create your views here.

def index_view(request):
  template = loader.get_template('index.html')
  return HttpResponse(template.render())



@csrf_exempt
def generate_charts(request):
    uploaded_file = request.FILES['file']
    try:
      df = pd.read_csv(uploaded_file)
      df.to_pickle(df_file_path)
    except Exception as e:
      return f"Erro ao processar o arquivo: {e}", 500
    
    charts = {
        'genre_popularity': create_genre_popularity_chart(df),
        'developer_rating_chart' : create_developer_rating_chart(df),
        'genre_distribution_developer' : create_genre_distribution_developer(df),
        'games_developer' : create_games_developer(df),
        'games_reviews' : create_games_reviews(df)
    }

    os.makedirs(media_dir, exist_ok=True)
    response = {}
    
    for name, figure in charts.items():
      filename = f'{name}.jpg'
      output_file = os.path.join(media_dir, filename)
      figure.savefig(output_file, format='jpg', dpi=300, bbox_inches='tight')
      plt.close(figure)

      response[name] = filename


    return JsonResponse(response)

# @csrf_exempt
def get_select_values(request):
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
    'genres': unique_genres.tolist(),
    'details': unique_details.tolist(),
  }

  return JsonResponse(response)
    

def serve_media_file(request, filename):
    # Construct the full file path
    file_path = os.path.join(media_dir, filename)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise Http404("File not found")

    # Serve the file
    response = FileResponse(open(file_path, 'rb'))
    return response