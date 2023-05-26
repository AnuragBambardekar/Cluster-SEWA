"""clusterapp URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
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
from django.conf.urls.static import static
from django.conf import settings
from django.urls import include
from .views import home,post,signupPage,loginPage,dashboard,aboutPage
from service_1 import views

urlpatterns = [
    path('', views.index),
    path('home', views.home),
    path('blog/<slug:url>',post), # service1/blog/clustering
    path('signup',views.signupPage, name='signup'),
    path('login',views.loginPage, name='login'),
    path('about',views.aboutPage, name='about'),
    path('logout',views.logoutPage, name='logout'),
    path('dashboard',views.dashboard, name='dashboard'),
    path('service2',views.services, name='services'),
]