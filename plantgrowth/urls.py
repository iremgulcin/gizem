from django.contrib import admin
from django.urls import path,re_path,include
from . import views
from .models import rf_model_ec_transformed 


urlpatterns = [
    re_path("admin/", admin.site.urls),
    re_path(r'predict-ec/', views.predict_ec, name='predict_ec'),
    re_path(r'predict-nutrients/', views.predict_nutrients, name='predict_nutrients'),
    re_path(r'predict-water/', views.predict_water, name='predict_water'),
    re_path(r'predict-n2o/', views.predict_n2o, name='predict_n2o'),
    re_path(r'/predict-co2/', views.predict_co2, name='predict_co2'),
    re_path(r'/predict-ch4/', views.predict_ch4, name='predict_ch4'),
    re_path(r'/predict-substrate/', views.predict_substrate, name='predict_substrate'),

]

