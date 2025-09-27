"""
URL configuration for DiabetesPrediction project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
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
from django.urls import path, include
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),

    # Public landing page at root; authenticated users are redirected to home inside view
    path('', views.index, name='index'),

    # App pages
    path('home/', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('predict/result', views.result, name='result'),
    path('predict/download', views.download_result_pdf, name='download_result_pdf'),
    path('about/', views.about, name='about'),
    path('performance/', views.performance, name='performance'),
    path('api/performance-analysis/', views.performance_analysis_api, name='performance_analysis_api'),
    path('api/nearby-hospitals/', views.nearby_hospitals_api, name='nearby_hospitals_api'),

    # Accounts app URLs
    path('accounts/', include('accounts.urls')),
]

