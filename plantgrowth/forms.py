from django import forms

class PredictionForm(forms.Form):
    days_since_planting = forms.IntegerField(label='Ekimden Sonra Geçen Gün Sayısı')
    T_avg = forms.FloatField(label='Ortalama Sıcaklık (T_avg)')
    CO2_avg = forms.FloatField(label='Ortalama CO2 Seviyesi (CO2_avg)')





class EmissionPredictionForm(forms.Form):
    days_since_planting = forms.IntegerField(label='Ekimden Sonra Geçen Gün Sayısı')
    T_avg = forms.FloatField(label='Ortalama Sıcaklık (T_avg)')
    CO2_avg = forms.FloatField(label='Ortalama CO2 Seviyesi (CO2_avg)')
    Na = forms.FloatField(label='Na Seviyesi (mg)')
    K = forms.FloatField(label='K Seviyesi (mg)')
    Mg = forms.FloatField(label='Mg Seviyesi (mg)')
    Ca = forms.FloatField(label='Ca Seviyesi (mg)')
    N = forms.FloatField(label='N Seviyesi (%)')


    