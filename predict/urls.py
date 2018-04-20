from . import views
from django.urls import path

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.inappropriate_access),
    path('ic/', views.image_classification),
] + static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
