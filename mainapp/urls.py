from django.contrib import admin
from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.intro, name='intro'),
    path('index', views.index, name='index'),
    path('output',views.output,name='output'),
    path('predicted',views.predicted,name='predicted'),
    path('sendfile',views.sendfile,name='sendfile'),
    path('login',views.login,name='login'),
    path('register', views.register, name='register'),
    path('intro',views.intro, name='intro')

]

urlpatterns += static(settings.MEDIA_URL, document_root =settings.MEDIA_ROOT)
