from . import views
from django.urls import path


urlpatterns = [
    path('', views.maindoor, name='maindoor'),
    path('result/', views.result),
]