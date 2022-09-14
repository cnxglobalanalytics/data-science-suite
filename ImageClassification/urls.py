"""ImageClassification URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
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
from AppImageClassification import views
from django.conf import settings

admin.site.site_header = "Concentrix Image Classification"
admin.site.site_title = "Concentrix Image Classification Portal"
admin.site.index_title = "Welcome to Concentrix Image Classification Portal"

urlpatterns = [
                  path('admin/', admin.site.urls),
                  path('', views.dashboard, name='home'),
                  path('nlp/', views.nlp_view, name='nlp'),
                  path('quant_analytics/', views.quant_view, name='quant_analytics'),
                  path('fraud_analytics/', views.fraud_view, name='fraud_analytics'),
                  path('dashboard/', views.dashboard, name='dashboard'),
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
              
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)