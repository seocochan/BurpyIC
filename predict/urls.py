from . import views
from django.urls import path

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.inappropriate_access),
    path('ic/', views.image_classification),
    path('train_data/', views.on_recommend_train_data),
    path('train/', views.on_recommend_train),
    path('predict_data/', views.on_recommend_predict_data),
    path('predict/', views.on_recommend_predict),
    path('predict_result/', views.on_recommend_predict_result),
] + static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
