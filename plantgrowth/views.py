from django.shortcuts import render
from . import models  # Kullanıcı girdilerini almak için oluşturduğunuz form
import pandas as pd
from .models import rf_model_ec_transformed  # Modelinizi Django projesine dahil ettiğiniz yer
from django.http import HttpResponse
from .models import rf_model_new_water  # Modelinizi import edin
from .models import clf  # Eğitilmiş modelinizi import edin
import joblib
from django.shortcuts import render
from .forms import PredictionForm  # Önceden oluşturduğunuz form
from .forms import  EmissionPredictionForm
from .models import rf_model_ec_transformed  # Modelinizi import edin
import pandas as pd
from django.http import JsonResponse
from .forms import EmissionPredictionForm  # Önceden oluşturduğunuz form
from .models import rf_model_n2o  # Modelinizi import edin
import numpy as np


from .models import rf_model_nutrients  # Modelinizi import edin

def predict_ec(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            days_since_planting = form.cleaned_data['days_since_planting']
            T_avg = form.cleaned_data['T_avg']
            CO2_avg = form.cleaned_data['CO2_avg']

            # Verileri DataFrame olarak hazırla
            data = {
                'days_since_planting': [days_since_planting],
                'T_avg_log': [np.log(T_avg + 1)],
                'CO2_avg_log': [np.log(CO2_avg + 1)],
                'days_since_planting_squared': [days_since_planting ** 2]
            }
            df = pd.DataFrame(data)

            # Model tahminini yap
            prediction = rf_model_ec_transformed.predict(df)

            # Tahmin sonucunu döndür
            return JsonResponse({'Optimize EC': prediction[0]})
    else:
        form = PredictionForm()

    return render(request, 'predict_form.html', {'form': form})




def predict_nutrients(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Formdan alınan verilerle tahmin yap
            cleaned_data = form.cleaned_data
            X_new = pd.DataFrame([[
                cleaned_data['days_since_planting'],
                np.log(cleaned_data['T_avg'] + 1),
                np.log(cleaned_data['CO2_avg'] + 1),
                cleaned_data['days_since_planting'] ** 2
            ]], columns=['days_since_planting', 'T_avg_log', 'CO2_avg_log', 'days_since_planting_squared'])
            
            # Tahmin yap
            prediction = rf_model_nutrients.predict(X_new)
            
            # Tahmin sonucunu context'e ekleyip template'e gönder
            return JsonResponse({'Optimize besin degerleri (Na, K, Mg, Ca, N)': prediction[0].tolist()})
    else:
        form = PredictionForm()

    return render(request, 'predict_nutrients_result.html', {'form': form})




def predict_water(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Formdan verileri al
            cleaned_data = form.cleaned_data
            X_predict = pd.DataFrame([[
                cleaned_data['days_since_planting'],
                np.log(cleaned_data['T_avg'] + 1),
                np.log(cleaned_data['CO2_avg'] + 1),
                cleaned_data['days_since_planting'] ** 2
            ]], columns=['days_since_planting', 'T_avg_log', 'CO2_avg_log', 'days_since_planting_squared'])
            
            # Tahmin yap
            prediction = rf_model_new_water.predict(X_predict)
            

             # Tahmin sonucunu göster
            return JsonResponse({'Optimize su miktari': prediction[0]})
    else:
        form = PredictionForm()

    return render(request, 'predict_water_result.html', {'form': form})

           


def predict_n2o(request):
    if request.method == 'POST':
        form = EmissionPredictionForm(request.POST)
        if form.is_valid():
            # Formdan verileri al ve DataFrame hazırla
            cleaned_data = form.cleaned_data
            X_new = pd.DataFrame([[
                cleaned_data['days_since_planting'],
                np.log(cleaned_data['T_avg'] + 1),
                np.log(cleaned_data['CO2_avg'] + 1),
                cleaned_data['days_since_planting'] ** 2,
                cleaned_data['Na'],
                cleaned_data['K'],
                cleaned_data['Mg'],
                cleaned_data['Ca'],
                cleaned_data['N_percnt']
            ]], columns=['days_since_planting', 'T_avg_log', 'CO2_avg_log', 'EC_limit', 'Na', 'K', 'Mg', 'Ca', 'N_percnt'])
            
            # Tahmin yap
            prediction = rf_model_n2o.predict(X_new)
            
            return JsonResponse({'N2O Emisyonu': prediction[0]})
    else:
        form = EmissionPredictionForm()

    return render(request, 'predict_n2o_result.html', {'form': form})





def predict_co2(request):
    if request.method == 'POST':
        form = EmissionPredictionForm(request.POST)
        if form.is_valid():
            model = joblib.load('trained_models/best_rf_model_co2.pkl')  # Modeli yükle
            # Form verilerini hazırla
            cleaned_data = form.cleaned_data
            X_new = pd.DataFrame([[
                cleaned_data['days_since_planting'],
                np.log(cleaned_data['T_avg'] + 1),
                np.log(cleaned_data['CO2_avg'] + 1),
                cleaned_data['days_since_planting'] ** 2,
                cleaned_data['Na'],
                cleaned_data['K'],
                cleaned_data['Mg'],
                cleaned_data['Ca'],
                cleaned_data['N_percnt']
            ]], columns=['days_since_planting', 'T_avg_log', 'CO2_avg_log', 'EC_limit', 'Na', 'K', 'Mg', 'Ca', 'N_percnt'])
            
            # Tahmin yap
            prediction = model.predict(X_new)
            # Sonuçları göster
            return JsonResponse({'CO2 Emisyonu': prediction[0]})
    else:
        form = EmissionPredictionForm()

    return render(request, 'predict_co2_result.html', {'form': form})




def predict_ch4(request):
    if request.method == 'POST':
        form = EmissionPredictionForm(request.POST)
        if form.is_valid():
            model = joblib.load('trained_models/best_rf_model_ch4_reduced.pkl')  # Modeli yükle
            cleaned_data = form.cleaned_data
            X_new = pd.DataFrame([[
                cleaned_data['days_since_planting'],
                np.log(cleaned_data['T_avg'] + 1),
                np.log(cleaned_data['CO2_avg'] + 1),
                cleaned_data['days_since_planting'] ** 2,
                cleaned_data['Na'],
                cleaned_data['K'],
                cleaned_data['Mg'],
                cleaned_data['Ca'],
                cleaned_data['N_percnt']
            ]], columns=['days_since_planting', 'T_avg_log', 'CO2_avg_log', 'EC_limit', 'Na', 'K', 'Mg', 'Ca', 'N_percnt'])
            prediction = model.predict(X_new)
            return JsonResponse({'CH4 Emisyonu': prediction[0]})
    else:
        form = EmissionPredictionForm()

    return render(request, 'predict_ch4_result.html', {'form': form})


def predict_substrate(request):
    if request.method == 'POST':
        form = EmissionPredictionForm(request.POST)
        if form.is_valid():
            # Formdan verileri al
            cleaned_data = form.cleaned_data
            X_new = pd.DataFrame([[
                cleaned_data['days_since_planting'],
                np.log(cleaned_data['T_avg'] + 1),
                np.log(cleaned_data['CO2_avg'] + 1),
                cleaned_data['days_since_planting'] ** 2,
                cleaned_data['Na'],
                cleaned_data['K'],
                cleaned_data['Mg'],
                cleaned_data['Ca'],
                cleaned_data['N_percnt']
            ]], columns=['days_since_planting', 'T_avg_log', 'CO2_avg_log', 'EC_limit', 'Na', 'K', 'Mg', 'Ca', 'N_percnt'])
            
            # Tahmin yap
            prediction = clf.predict(X_new)
            
            # Sonucu göster
            return JsonResponse({'Optimize Substrat': prediction[0]})
    else:
        form = EmissionPredictionForm()

    return render(request, 'predict_substrat_result.html', {'form': form})

