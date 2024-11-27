import re
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, StandardScaler
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, mean_squared_error
import xgboost as xgb
import numpy as np
import os

from app.format_dataset import format_dataset 

data_dir = "data"

def train(df):
  print('Starting train...')

  # fazendo copia para fazer ajustes sem alterar o original
  data_df = (
     df[['name', 'genre_list', 'detail_list', 'price', 'percent_positive']]
    .dropna(subset=['percent_positive', 'price'])
  )

  mlb_genre = MultiLabelBinarizer()
  mlb_details = MultiLabelBinarizer()

  price_features = data_df[['price']]
  # faz aplicar transformadores para os generos e detalhes
  genre_features = mlb_genre.fit_transform(data_df['genre_list'])
  details_features = mlb_details.fit_transform(data_df['detail_list'])

  # transforma a lista de generos e detalhes em uma coluna para valor possível, com 0/1 para indicar se está presente
  genre_df = pd.DataFrame(
    genre_features, 
    columns=[f'genre_{i}' for i in range(genre_features.shape[1])]
  )
  
  details_df = pd.DataFrame(
    details_features, 
    columns=[f'detail_{i}' for i in range(details_features.shape[1])]
  )

  # cria df novo com as caracteristicas
  X = pd.concat([genre_df, details_df, price_features], axis=1)
  X = X.fillna(0)
  
  # target sendo a % positiva
  y = data_df[['percent_positive']]

  # há o caso de a qtd de dados de caracteristicas ser diferente que a de targets (nao consegui entender pq)
  print(f"Shape of X: {X.shape}")
  print(f"Shape of y: {y.shape}")

  # faz interseção entre X e y, e remove linhas que existem valores só de um lado
  combined = pd.concat([X, y], axis=1)
  combined = combined.dropna()
    
  X = combined[X.columns]
  y = combined['percent_positive']

  # combined.to_excel('combined.xlsx', index=False)

  scaler = MinMaxScaler()
  X = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns  # Preserve column names after scaling
  )

  print(X.size)
  

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

  model = xgb.XGBRegressor(
      colsample_bytree=0.8,
      learning_rate=0.05,
      max_depth=5,
      n_estimators=150,
      subsample=0.8,
      random_state=42,
      objective='reg:squarederror',
  )

  # model = GradientBoostingRegressor(
  #     n_estimators=80,
  #     max_depth=6,
  #     learning_rate=0.01,
  #     random_state=200
  # )

  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)

  print(y_test)
  print(y_pred)

  # metricas
  mse = mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mse)
  mae = mean_absolute_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  print('\nRegression Metrics:')
  print(f'Mean Squared Error: {mse:.4f}')
  print(f'Root Mean Squared Error: {rmse:.4f}')
  print(f'Mean Absolute Error: {mae:.4f}')
  print(f'R² Score: {r2:.4f}')

  # caracteristicas mais importantes
  importance_df = pd.DataFrame({
      'Caracteristica': X.columns,
      'Importancia': model.feature_importances_
  }).sort_values('Importancia', ascending=False)
  
  print('\nCaracteristicas mais importantes:')
  print(importance_df.head(10))

  # salva dados do modelo
  joblib.dump(model, get_data_path('review_prediction_model.pkl'))
  joblib.dump(scaler, get_data_path('scaler.pkl'))
  joblib.dump(mlb_genre, get_data_path('mlb_genre.pkl'))
  joblib.dump(mlb_details, get_data_path('mlb_details.pkl'))

  print(y_pred)

  return {
     'metrics': {
          'meanSquaredError': mse,
          'rootMeanSquaredError': rmse,
          'meanAbsoluteError': mae,
          'r2Score': r2,
     }
  }

def get_data_path(filename):
   return os.path.join(data_dir, filename)

def predict_positive_review(price, detail, genre):
  # print(get_data_path('review_prediction_model.pkl'))
  # carrega o modelo dos arquivos
  model = joblib.load(get_data_path('review_prediction_model.pkl'))
  scaler = joblib.load(get_data_path('scaler.pkl'))
  mlb_genre = joblib.load(get_data_path('mlb_genre.pkl'))
  mlb_details = joblib.load(get_data_path('mlb_details.pkl'))

  genre_list = genre.split(',')
  detail_list = detail.split(',')

  # aplica transoformação configurada anteriormente
  genre_features = mlb_genre.transform([genre_list])
  details_features = mlb_details.transform([detail_list])
  price_feature = np.array([[price]])

  X_input = np.concatenate([genre_features, details_features, price_feature], axis=1)

  print(X_input)

  X_input = scaler.transform(X_input)

  # predição
  predicted_percent = model.predict(X_input)[0]

  # converte em porcentagem
  predicted_percent = predicted_percent * 100.0

  print(predicted_percent)

  return predicted_percent.item()

def is_trained():
   return os.path.exists(get_data_path('review_prediction_model.pkl'))