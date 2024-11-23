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

data_dir = "data"

def extract_reviews_info(reviews_text):
    # print(reviews_text)
    match = re.search(r"(\d+(?:,\d+)*)", reviews_text)
    num_reviews = int(match.group(1).replace(",", "")) if match else 0
    match_percent = re.search(r"(\d+)%", reviews_text)
    percent_positive = int(match_percent.group(1)) if match_percent else 0
    return num_reviews, percent_positive

def extract_price(price_text):
   if (isinstance(price_text, str) and "free" in price_text.lower()):
      return 0.0
   
  #  print(price_text)
   try:
      return float(price_text.replace("$", ""))
   except:
      return None

def categorize_sentiment(percent):
  if percent >= 70:
      return 'positive'
  elif percent >= 40:
      return 'mixed'
  else:
      return 'negative'

def train(df):
  print('Starting train...')
  data_df = pd.DataFrame()

  # removendo linhas que o review ou preco tem nan
  data_df = df[['name', 'genre', 'game_details', 'original_price', 'all_reviews']].dropna(subset=['all_reviews', 'original_price'])

  data_df[['num_reviews', 'percent_positive']] = data_df['all_reviews'].apply(
    lambda x: pd.Series(extract_reviews_info(x))
  )

  # print(train_df['original_price'])

  data_df['genre_list'] = data_df['genre'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
  data_df['detail_list'] = data_df['game_details'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
  data_df['original_price'] = data_df['original_price']
  data_df['price'] = data_df['original_price'].apply(extract_price)

  # remove os preços invalidos (None), tipo demo etc
  data_df = data_df.dropna(subset=['price', 'num_reviews', 'percent_positive'])

  # data_df['sentiment'] = data_df['percent_positive'].apply(categorize_sentiment)

  # transformar numero de reviews em log  
  # data_df['num_reviews'] = np.log1p(data_df['num_reviews'])
  data_df['percent_positive'] = data_df['percent_positive'] / 100.0

  data_df.to_excel('output.xlsx', index=False)

  mlb_genre = MultiLabelBinarizer()
  mlb_details = MultiLabelBinarizer()

  genre_features = mlb_genre.fit_transform(data_df['genre_list'])
  details_features = mlb_details.fit_transform(data_df['detail_list'])
  price_features = data_df[['price']]


  genre_df = pd.DataFrame(
    genre_features, 
    columns=[f'genre_{i}' for i in range(genre_features.shape[1])]
  )
  
  details_df = pd.DataFrame(
    details_features, 
    columns=[f'detail_{i}' for i in range(details_features.shape[1])]
  )

  # Concatenate with named columns
  X = pd.concat([genre_df, details_df, price_features], axis=1)
  X = X.fillna(0)
  
  # y = data_df[['sentiment']]
  y = data_df[['percent_positive']]

  print(f"Shape of X: {X.shape}")
  print(f"Shape of y: {y.shape}")

  combined = pd.concat([X, y], axis=1)
  combined = combined.dropna()
    
  X = combined[X.columns]
  y = combined['percent_positive']

  combined.to_excel('combined.xlsx', index=False)

  scaler = MinMaxScaler()
  X = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns  # Preserve column names after scaling
  )

  print(f"Shape of X: {X.shape}")
  print(f"Shape of y: {y.shape}")

  print(X.size)
  

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  param_grid = {
      'n_estimators': [50, 100, 150],
      'max_depth': [3, 5, 7],
      'learning_rate': [0.01, 0.05, 0.1],
      'min_samples_split': [2, 5, 10],
  }
  # grid_search = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
  # grid_search.fit(X_train, y_train)
  # print(f'Best parameters: {grid_search.best_params_}')
 
  model = xgb.XGBRegressor(
      colsample_bytree=0.8,
      learning_rate=0.05,
      max_depth=5,
      n_estimators=150,
      subsample=0.8,
      random_state=42,
      objective='reg:squarederror'  # Squared error for regression
  )


  # param_grid = {
  #       'n_estimators': [50, 100, 150],
  #       'max_depth': [3, 5, 7],
  #       'learning_rate': [0.01, 0.05, 0.1],
  #       'subsample': [0.8, 1.0],  # Fraction of samples to use for training
  #       'colsample_bytree': [0.8, 1.0]  # Fraction of features to use for each tree
  #   }

  # grid_search = GridSearchCV(
  #     model,
  #     param_grid,
  #     scoring='neg_mean_squared_error',
  #     cv=5,
  #     n_jobs=-1
  # )
  # grid_search.fit(X_train, y_train)

  # print(f'Best parameters: {grid_search.best_params_}')
  # model = grid_search.best_estimator_

  # model = GradientBoostingRegressor(
  #     n_estimators=80,
  #     max_depth=6,
  #     learning_rate=0.01,
  #     random_state=200
  # )

  # model = RandomForestClassifier(
  #     n_estimators=100,
  #     max_depth=10,
  #     random_state=42
  # )

  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)

  print(y_test)
  print(y_pred)

  mse = mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mse)
  mae = mean_absolute_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  print('\nRegression Metrics:')
  print(f'Mean Squared Error: {mse:.4f}')
  print(f'Root Mean Squared Error: {rmse:.4f}')
  print(f'Mean Absolute Error: {mae:.4f}')
  print(f'R² Score: {r2:.4f}')

  # Print example predictions
  print('\nExample predictions (in percentages):')
  sample_indices = np.random.choice(len(y_test), 5, replace=False)
  for idx in sample_indices:
      print(f'Actual: {y_test.iloc[idx]:.1f}%, Predicted: {y_pred[idx]:.1f}%')

  # Feature importance
  importance_df = pd.DataFrame({
      'Feature': X.columns,
      'Importance': model.feature_importances_
  }).sort_values('Importance', ascending=False)
  
  print('\nTop 10 most important features:')
  print(importance_df.head(10))

  joblib.dump(model, get_data_path('review_prediction_model.pkl'))
  joblib.dump(scaler, get_data_path('scaler.pkl'))
  joblib.dump(mlb_genre, get_data_path('mlb_genre.pkl'))
  joblib.dump(mlb_details, get_data_path('mlb_details.pkl'))

  print(y_pred)

def get_data_path(filename):
   return os.path.join(data_dir, filename)

def predict_positive_review(price, detail, genre):
    # Load the saved model and preprocessors
    model = joblib.load(get_data_path('review_prediction_model.pkl'))
    scaler = joblib.load(get_data_path('scaler.pkl'))
    mlb_genre = joblib.load(get_data_path('mlb_genre.pkl'))
    mlb_details = joblib.load(get_data_path('mlb_details.pkl'))

    # Prepare the input data
    genre_list = genre.split(',')
    detail_list = detail.split(',')

    genre_features = mlb_genre.transform([genre_list])
    details_features = mlb_details.transform([detail_list])
    price_feature = np.array([[price]])

    # Combine the features
    X_input = np.concatenate([genre_features, details_features, price_feature], axis=1)

    print(X_input)

    # Scale the input features
    X_input = scaler.transform(X_input)

    # Predict the percent of positive reviews
    predicted_percent = model.predict(X_input)[0]

    # Convert the prediction back to percentage
    predicted_percent = predicted_percent * 100.0

    print(predicted_percent)

    return predicted_percent.item()

def is_trained():
   return os.path.exists(get_data_path('review_prediction_model.pkl'))