"""
URL configuration for steam_dataset_analytics project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from app import views
from app.views import get_train_values, index_view, serve_media_file, train_view, predict_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('generate_charts/', views.generate_charts),
    path('media/<str:filename>', serve_media_file),
    path('train_values', get_train_values),
    path('train', train_view),
    path('predict', predict_view),
    path('', index_view),
]
