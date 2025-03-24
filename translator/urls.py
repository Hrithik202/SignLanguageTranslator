from django.urls import path
from django.conf import settings  
from django.conf.urls.static import static  
from . import views
from .views import camera_feed

urlpatterns=[
       path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('camera_feed/', views.camera_feed, name='camera_feed'),
    path('process_gesture/', views.process_gesture, name='process_gesture'),
]


if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)