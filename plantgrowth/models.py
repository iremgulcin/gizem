
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

pd.set_option('display.max_columns', None)  # Tüm sütunları göster



def update_df_values(df, begin_date, end_date):
    # Belirli bir başlangıç ve bitiş tarihi arasında olan ve değeri 5 olan kayıtları filtreleyelim
    mask = (df['Date'] >= begin_date) & (df['Date'] <= end_date) & (df['EC_limit'] == 5)

    # Bu koşulları sağlayan gün sayısını hesaplayalım
    days = np.arange(mask.sum())

    # Lineer olarak artan değerler oluşturalım, son değer 5 olacak şekilde
    if days.size > 0 and days.max() != 0:
        linear_values = 5 * days / days.max()
    else:
        # days boyutu 0 veya days.max() 0 ise, mask koşulunu sağlayan tüm kayıtlar için 0 değerini kullan
        linear_values = np.zeros(mask.sum())

    # linear_values boyutunu kontrol edin ve mask koşulunu sağlayan kayıtları güncelleyin
    df.loc[mask, 'EC_limit'] = linear_values[:mask.sum()] if mask.sum() != 0 else df.loc[mask, 'EC_limit']
    
    return df




def update_df_values2(df, begin_date, end_date):
    # Belirli bir başlangıç ve bitiş tarihi arasında olan ve değeri 5 olan kayıtları filtreleyelim
    mask = (df['Date'] >= begin_date) & (df['Date'] <= end_date) & (df['EC_limit'] == 7.5)

    # Bu koşulları sağlayan gün sayısını hesaplayalım
    days = np.arange(mask.sum())

    # Lineer olarak artan değerler oluşturalım, son değer 5 olacak şekilde
    if days.size > 0 and days.max() != 0:
        linear_values = 7.5 * days / days.max()
    else:
        # days boyutu 0 veya days.max() 0 ise, mask koşulunu sağlayan tüm kayıtlar için 0 değerini kullan
        linear_values = np.zeros(mask.sum())

    # linear_values boyutunu kontrol edin ve mask koşulunu sağlayan kayıtları güncelleyin
    df.loc[mask, 'EC_limit'] = linear_values[:mask.sum()] if mask.sum() != 0 else df.loc[mask, 'EC_limit']
    
    return df




def update_df_values3(df, begin_date, end_date):
    # Belirli bir başlangıç ve bitiş tarihi arasında olan ve değeri 5 olan kayıtları filtreleyelim
    mask = (df['Date'] >= begin_date) & (df['Date'] <= end_date) & (df['EC_limit'] == 10)

    # Bu koşulları sağlayan gün sayısını hesaplayalım
    days = np.arange(mask.sum())

    # Lineer olarak artan değerler oluşturalım, son değer 5 olacak şekilde
    if days.size > 0 and days.max() != 0:
        linear_values = 10 * days / days.max()
    else:
        # days boyutu 0 veya days.max() 0 ise, mask koşulunu sağlayan tüm kayıtlar için 0 değerini kullan
        linear_values = np.zeros(mask.sum())

    # linear_values boyutunu kontrol edin ve mask koşulunu sağlayan kayıtları güncelleyin
    df.loc[mask, 'EC_limit'] = linear_values[:mask.sum()] if mask.sum() != 0 else df.loc[mask, 'EC_limit']
    
    return df


data1_1=pd.read_excel(r"C:\Users\Gizem\Desktop\IC\1 - Dry matter production.xlsx")

data1_1c=data1_1.copy()

data1_1c.rename(columns ={'EC limit':'EC_limit'}, inplace=True)

data1_1c.rename(columns ={'Organ harvested':'Organ'},inplace=True)

data1_1c["Organ"]=1

secilen_sutunlar = ['Date', 'EC_limit', 'Organ', 'Leaves']  # İstediğiniz sütun isimlerini buraya yazın
yeni_df = data1_1c[secilen_sutunlar]

grouped_df1_1a = yeni_df

grouped_df1_1a = grouped_df1_1a.sort_values(by=['Date'])

grouped_df1_1a["Organ"]=3

grouped_df1_1a["Date"]=grouped_df1_1a.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45,"12_March":48, "13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })

baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df1_1a['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df1_1a['Date'] - 1, unit='D')

grouped_df1_1a['Date'] = grouped_df1_1a['Date'].astype(float)
grouped_df1_1a['Organ'] = grouped_df1_1a['Organ'].astype(float)
grouped_df1_1a['EC_limit'] = grouped_df1_1a['EC_limit'].astype(float)

grouped_df1_1a.rename(columns ={'Leaves':'Dry_matter_perplant'},inplace=True)

grouped_df1_1a = update_df_values(grouped_df1_1a, 1, 92)
grouped_df1_1a = update_df_values(grouped_df1_1a, 93, 106)
grouped_df1_1a = update_df_values(grouped_df1_1a, 107, 118)
grouped_df1_1a = update_df_values(grouped_df1_1a, 119, 131)
grouped_df1_1a = update_df_values(grouped_df1_1a, 132, 145)
grouped_df1_1a = update_df_values(grouped_df1_1a, 146, 171)
grouped_df1_1a = update_df_values2(grouped_df1_1a, 1, 106)
grouped_df1_1a = update_df_values2(grouped_df1_1a, 107, 140)
grouped_df1_1a = update_df_values2(grouped_df1_1a, 141, 171)
grouped_df1_1a = update_df_values3(grouped_df1_1a, 1, 118)
grouped_df1_1a = update_df_values3(grouped_df1_1a, 119, 171)


filtered_df = grouped_df1_1a[grouped_df1_1a['EC_limit'] <= 5]

secilen_sutunlar2 = ['Date', 'EC_limit', 'Organ', 'Stems']  # İstediğiniz sütun isimlerini buraya yazın
yeni_df2 = data1_1c[secilen_sutunlar2]


grouped_df1_1b=yeni_df2.sort_values(by=['EC_limit','Date'])

grouped_df1_1b = yeni_df2.groupby(['Date', 'EC_limit']).mean().reset_index()

grouped_df1_1b["Organ"]=4


grouped_df1_1b["Date"]=grouped_df1_1b.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45, "12_March":48,"13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })


baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df1_1b['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df1_1b['Date'] - 1, unit='D')


grouped_df1_1b['Date'] = grouped_df1_1b['Date'].astype(float)
grouped_df1_1b['Organ'] = grouped_df1_1b['Organ'].astype(float)
grouped_df1_1b['EC_limit'] = grouped_df1_1b['EC_limit'].astype(float)


grouped_df1_1b.rename(columns ={'Stems':'Dry_matter_perplant'},inplace=True)

grouped_df1_1b = update_df_values(grouped_df1_1b, 1, 92)
grouped_df1_1b = update_df_values(grouped_df1_1b, 93, 106)
grouped_df1_1b = update_df_values(grouped_df1_1b, 107, 118)
grouped_df1_1b = update_df_values(grouped_df1_1b, 119, 131)
grouped_df1_1b = update_df_values(grouped_df1_1b, 132, 145)
grouped_df1_1b = update_df_values(grouped_df1_1b, 146, 171)
grouped_df1_1b = update_df_values2(grouped_df1_1b, 1, 106)
grouped_df1_1b = update_df_values2(grouped_df1_1b, 107, 140)
grouped_df1_1b = update_df_values2(grouped_df1_1b, 141, 171)
grouped_df1_1b = update_df_values3(grouped_df1_1b, 1, 118)
grouped_df1_1b = update_df_values3(grouped_df1_1b, 119, 171)


secilen_sutunlar3 = ['Date', 'EC_limit', 'Organ', 'Fruits']  # İstediğiniz sütun isimlerini buraya yazın
yeni_df3= data1_1c[secilen_sutunlar3]


grouped_df1_1c = yeni_df3


grouped_df1_1c = grouped_df1_1c.sort_values(by=['Date','EC_limit'])

grouped_df1_1c["Organ"]=2


grouped_df1_1c["Date"]=grouped_df1_1c.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45, "12_March":48,"13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })

baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df1_1c['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df1_1c['Date'] - 1, unit='D')

grouped_df1_1c['Date'] = grouped_df1_1c['Date'].astype(float)
grouped_df1_1c['Organ'] = grouped_df1_1c['Organ'].astype(float)
grouped_df1_1c['EC_limit'] = grouped_df1_1c['EC_limit'].astype(float)

grouped_df1_1c.rename(columns ={'Fruits':'Dry_matter_perplant'},inplace=True)


grouped_df1_1c = update_df_values(grouped_df1_1c, 1, 92)
grouped_df1_1c = update_df_values(grouped_df1_1c, 93, 106)
grouped_df1_1c = update_df_values(grouped_df1_1c, 107, 118)
grouped_df1_1c = update_df_values(grouped_df1_1c, 119, 131)
grouped_df1_1c = update_df_values(grouped_df1_1c, 132, 145)
grouped_df1_1c = update_df_values(grouped_df1_1c, 146, 171)
grouped_df1_1c = update_df_values2(grouped_df1_1c, 1, 106)
grouped_df1_1c = update_df_values2(grouped_df1_1c, 107, 140)
grouped_df1_1c = update_df_values2(grouped_df1_1c, 141, 171)
grouped_df1_1c = update_df_values3(grouped_df1_1c, 1, 118)
grouped_df1_1c = update_df_values3(grouped_df1_1c, 119, 171)


grouped_df1_1c

secilen_sutunlar4 = ['Date', 'EC_limit', 'Organ', 'Roots']  # İstediğiniz sütun isimlerini buraya yazın
yeni_df4= data1_1c[secilen_sutunlar4]

grouped_df1_1d = yeni_df4

grouped_df1_1d = grouped_df1_1d.sort_values(by=['Date','EC_limit'])



grouped_df1_1d["Organ"]=5


grouped_df1_1d["Date"]=grouped_df1_1d.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45, "12_March":48,"13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })

baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df1_1d['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df1_1d['Date'] - 1, unit='D')


grouped_df1_1d['Date'] = grouped_df1_1d['Date'].astype(float)
grouped_df1_1d['Organ'] = grouped_df1_1d['Organ'].astype(float)
grouped_df1_1d['EC_limit'] = grouped_df1_1d['EC_limit'].astype(float)

grouped_df1_1d.rename(columns ={'Roots':'Dry_matter_perplant'},inplace=True)


grouped_df1_1d = update_df_values(grouped_df1_1d, 1, 92)
grouped_df1_1d = update_df_values(grouped_df1_1d, 93, 106)
grouped_df1_1d = update_df_values(grouped_df1_1d, 107, 118)
grouped_df1_1d = update_df_values(grouped_df1_1d, 119, 131)
grouped_df1_1d = update_df_values(grouped_df1_1d, 132, 145)
grouped_df1_1d = update_df_values(grouped_df1_1d, 146, 171)
grouped_df1_1d = update_df_values2(grouped_df1_1d, 1, 106)
grouped_df1_1d = update_df_values2(grouped_df1_1d, 107, 140)
grouped_df1_1d = update_df_values2(grouped_df1_1d, 141, 171)
grouped_df1_1d = update_df_values3(grouped_df1_1d, 1, 118)
grouped_df1_1d = update_df_values3(grouped_df1_1d, 119, 171)



data1_2=pd.read_excel(r"C:\Users\Gizem\Desktop\IC\1 - Dry matter production-2.xlsx")


grouped_df1_2=data1_2.copy()

grouped_df1_2.rename(columns ={'Organ harvested':'Organ'},inplace=True)

grouped_df1_2['Organ'] = 2

grouped_df1_2= grouped_df1_2.drop('Replication', axis=1)

grouped_df1_2.rename(columns ={'Dry_matter':'Dry_matter_perplant'},inplace=True)

grouped_df1_2.rename(columns ={'Trusses number':'Trusses_number'},inplace=True)

grouped_df1_2.rename(columns ={'EC limit':'EC_limit'}, inplace=True)

grouped_df1_2.rename(columns ={'Dry matter/truss (g)':'Dry_matter_pertruss'},inplace=True)

grouped_df1_2["Date"]=grouped_df1_2.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45, "12_March":48,"13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })

grouped_df1_2 = grouped_df1_2.sort_values(by=['Date','EC_limit'])

baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df1_2['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df1_2['Date'] - 1, unit='D')

grouped_df1_2['Date'] = grouped_df1_2['Date'].astype(float)
grouped_df1_2['Organ'] = grouped_df1_2['Organ'].astype(float)
grouped_df1_2['EC_limit'] = grouped_df1_2['EC_limit'].astype(float)

grouped_df1_2 = update_df_values(grouped_df1_2, 1, 92)
grouped_df1_2 = update_df_values(grouped_df1_2, 93, 106)
grouped_df1_2 = update_df_values(grouped_df1_2, 107, 118)
grouped_df1_2 = update_df_values(grouped_df1_2, 119, 131)
grouped_df1_2 = update_df_values(grouped_df1_2, 132, 145)
grouped_df1_2 = update_df_values(grouped_df1_2, 146, 171)
grouped_df1_2 = update_df_values2(grouped_df1_2, 1, 106)
grouped_df1_2 = update_df_values2(grouped_df1_2, 107, 140)
grouped_df1_2 = update_df_values2(grouped_df1_2, 141, 171)
grouped_df1_2 = update_df_values3(grouped_df1_2, 1, 118)
grouped_df1_2 = update_df_values3(grouped_df1_2, 119, 171)


data1_3=pd.read_excel(r"C:\Users\Gizem\Desktop\IC/1 - Dry matter production-3.xlsx")


grouped_df1_3=data1_3.copy()


grouped_df1_3.rename(columns ={'Organ harvested':'Organ'},inplace=True)

grouped_df1_3.rename(columns ={'EC limit':'EC_limit'}, inplace=True)

grouped_df1_3= grouped_df1_3.drop('Replication', axis=1)

grouped_df1_3['Organ'] = 3

grouped_df1_3.rename(columns ={'Dry matter/plant (g)':'Dry_matter_perplant'},inplace=True)


grouped_df1_3["Date"]=grouped_df1_3.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45, "12_March":48,"13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })


grouped_df1_3 = grouped_df1_3.sort_values(by=['Date','EC_limit'])

grouped_df1_3['Date'] = grouped_df1_3['Date'].astype(float)
grouped_df1_3['Organ'] = grouped_df1_3['Organ'].astype(float)
grouped_df1_3['EC_limit'] = grouped_df1_3['EC_limit'].astype(float)

baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df1_3['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df1_3['Date'] - 1, unit='D')


grouped_df1_3 = update_df_values(grouped_df1_3, 1, 92)
grouped_df1_3 = update_df_values(grouped_df1_3, 93, 106)
grouped_df1_3 = update_df_values(grouped_df1_3, 107, 118)
grouped_df1_3 = update_df_values(grouped_df1_3, 119, 131)
grouped_df1_3 = update_df_values(grouped_df1_3, 132, 145)
grouped_df1_3 = update_df_values(grouped_df1_3, 146, 171)
grouped_df1_3 = update_df_values2(grouped_df1_3, 1, 106)
grouped_df1_3 = update_df_values2(grouped_df1_3, 107, 140)
grouped_df1_3 = update_df_values2(grouped_df1_3, 141, 171)
grouped_df1_3 = update_df_values3(grouped_df1_3, 1, 118)
grouped_df1_3 = update_df_values3(grouped_df1_3, 119, 171)


data2_1=pd.read_excel(r"C:\Users\Gizem\Desktop\IC/2 - Nutrient solution consumption tomato-1.xlsx")


grouped_df2_1=data2_1.copy()

grouped_df2_1.rename(columns ={'EC limit':'EC_limit'}, inplace=True)

grouped_df2_1= grouped_df2_1.drop('Replication', axis=1)

grouped_df2_1.rename(columns ={'NS new/plant':'NS_new_perplant'},inplace=True)

grouped_df2_1.rename(columns ={'NS added/plant':'NS_added_perplant'},inplace=True)

grouped_df2_1.rename(columns ={'NS residual/plant':'NS_residual_perplant'},inplace=True)

grouped_df2_1.rename(columns ={'Day':'Date'},inplace=True)


grouped_df2_1["Date"]=grouped_df2_1.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45, "12_March":48,"13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })

grouped_df2_1 = grouped_df2_1.sort_values(by=['Date','EC_limit'])

grouped_df2_1['Organ'] = 1

grouped_df2_1['Date'] = grouped_df2_1['Date'].astype(float)
grouped_df2_1['Organ'] = grouped_df2_1['Organ'].astype(float)
grouped_df2_1['EC_limit'] = grouped_df2_1['EC_limit'].astype(float)

baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df2_1['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df2_1['Date'] - 1, unit='D')

grouped_df2_1 = update_df_values(grouped_df2_1, 1, 92)
grouped_df2_1 = update_df_values(grouped_df2_1, 93, 106)
grouped_df2_1 = update_df_values(grouped_df2_1, 107, 118)
grouped_df2_1 = update_df_values(grouped_df2_1, 119, 131)
grouped_df2_1 = update_df_values(grouped_df2_1, 132, 145)
grouped_df2_1 = update_df_values(grouped_df2_1, 146, 171)
grouped_df2_1 = update_df_values2(grouped_df2_1, 1, 106)
grouped_df2_1 = update_df_values2(grouped_df2_1, 107, 140)
grouped_df2_1= update_df_values2(grouped_df2_1, 141, 171)
grouped_df2_1 = update_df_values3(grouped_df2_1, 1, 118)
grouped_df2_1 = update_df_values3(grouped_df2_1, 119, 171)

grouped_df2_1

data3_1=pd.read_excel(r"C:\Users\Gizem\Desktop\IC/3-Leaves cations.xlsx")

grouped_df3_1=data3_1.copy()

grouped_df3_1.drop([0], axis=0, inplace=True)

grouped_df3_1.rename(columns ={'EC limit':'EC_limit'}, inplace=True)

grouped_df3_1= grouped_df3_1.drop('Replication', axis=1)

grouped_df3_1.rename(columns ={'Organ harvested':'Organ'},inplace=True)

grouped_df3_1['Organ'] = 3

grouped_df3_1["Date"]=grouped_df3_1.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45, "12_March":48,"13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })

grouped_df3_1 = grouped_df3_1.sort_values(by=['Date','EC_limit'])

grouped_df3_1['Date'] = grouped_df3_1['Date'].astype(float)
grouped_df3_1['Organ'] = grouped_df3_1['Organ'].astype(float)
grouped_df3_1['EC_limit'] = grouped_df3_1['EC_limit'].astype(float)

baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df3_1['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df3_1['Date'] - 1, unit='D')

grouped_df3_1 = update_df_values(grouped_df3_1, 1, 92)
grouped_df3_1 = update_df_values(grouped_df3_1, 93, 106)
grouped_df3_1 = update_df_values(grouped_df3_1, 107, 118)
grouped_df3_1 = update_df_values(grouped_df3_1, 119, 131)
grouped_df3_1 = update_df_values(grouped_df3_1, 132, 145)
grouped_df3_1 = update_df_values(grouped_df3_1, 146, 171)
grouped_df3_1 = update_df_values2(grouped_df3_1, 1, 106)
grouped_df3_1 = update_df_values2(grouped_df3_1, 107, 140)
grouped_df3_1= update_df_values2(grouped_df3_1, 141, 171)
grouped_df3_1 = update_df_values3(grouped_df3_1, 1, 118)
grouped_df3_1 = update_df_values3(grouped_df3_1, 119, 171)


data3_2=pd.read_excel(r"C:\Users\Gizem\Desktop\IC/3-leaves cations-2.xlsx")

grouped_df3_2=data3_2.copy()

grouped_df3_2.rename(columns ={'EC limit':'EC_limit'}, inplace=True)

grouped_df3_2= grouped_df3_2.drop('Replication', axis=1)

grouped_df3_2.rename(columns ={'Organ harvested':'Organ'},inplace=True)

grouped_df3_2['Organ'] = 3

grouped_df3_2["Date"]=grouped_df3_2.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45, "12_March":48,"13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })

grouped_df3_2 = grouped_df3_2.sort_values(by=['Date','EC_limit'])

grouped_df3_2['Date'] = grouped_df3_2['Date'].astype(float)
grouped_df3_2['Organ'] = grouped_df3_2['Organ'].astype(float)
grouped_df3_2['EC_limit'] = grouped_df3_2['EC_limit'].astype(float)

baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df3_2['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df3_2['Date'] - 1, unit='D')

grouped_df3_2 = update_df_values(grouped_df3_2, 1, 92)
grouped_df3_2 = update_df_values(grouped_df3_2, 93, 106)
grouped_df3_2 = update_df_values(grouped_df3_2, 107, 118)
grouped_df3_2 = update_df_values(grouped_df3_2, 119, 131)
grouped_df3_2 = update_df_values(grouped_df3_2, 132, 145)
grouped_df3_2 = update_df_values(grouped_df3_2, 146, 171)
grouped_df3_2 = update_df_values2(grouped_df3_2, 1, 106)
grouped_df3_2 = update_df_values2(grouped_df3_2, 107, 140)
grouped_df3_2= update_df_values2(grouped_df3_2, 141, 171)
grouped_df3_2 = update_df_values3(grouped_df3_2, 1, 118)
grouped_df3_2 = update_df_values3(grouped_df3_2, 119, 171)

data4_1=pd.read_excel(r"C:\Users\Gizem\Desktop\IC/4 - Stems cations-1.xlsx")

grouped_df4_1=data4_1.copy()

grouped_df4_1.rename(columns ={'EC limit':'EC_limit'}, inplace=True)

grouped_df4_1= grouped_df4_1.drop('Replication', axis=1)

grouped_df4_1.rename(columns ={'Organ harvested':'Organ'},inplace=True)

grouped_df4_1['Organ'] = 4

grouped_df4_1.drop([0,1,2,3,4,5,6,7,8], axis=0, inplace=True)

grouped_df4_1["Date"]=grouped_df4_1.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45, "12_March":48,"13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })

grouped_df4_1 = grouped_df4_1.sort_values(by=['Date','EC_limit'])

grouped_df4_1['Date'] = grouped_df4_1['Date'].astype(float)
grouped_df4_1['Organ'] = grouped_df4_1['Organ'].astype(float)
grouped_df4_1['EC_limit'] = grouped_df4_1['EC_limit'].astype(float)

baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df4_1['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df4_1['Date'] - 1, unit='D')

grouped_df4_1 = update_df_values(grouped_df4_1, 1, 92)
grouped_df4_1 = update_df_values(grouped_df4_1, 93, 106)
grouped_df4_1 = update_df_values(grouped_df4_1, 107, 118)
grouped_df4_1 = update_df_values(grouped_df4_1, 119, 131)
grouped_df4_1 = update_df_values(grouped_df4_1, 132, 145)
grouped_df4_1 = update_df_values(grouped_df4_1, 146, 171)
grouped_df4_1 = update_df_values2(grouped_df4_1, 1, 106)
grouped_df4_1 = update_df_values2(grouped_df4_1, 107, 140)
grouped_df4_1= update_df_values2(grouped_df4_1, 141, 171)
grouped_df4_1 = update_df_values3(grouped_df4_1, 1, 118)
grouped_df4_1 = update_df_values3(grouped_df4_1, 119, 171)

grouped_df4_1.isnull().sum()

data5_1=pd.read_excel(r"C:\Users\Gizem\Desktop\IC/5 - Fruit cations-1.xlsx")

grouped_df5_1=data5_1.copy()

grouped_df5_1.rename(columns ={'EC limit':'EC_limit'}, inplace=True)

grouped_df5_1= grouped_df5_1.drop('Replication', axis=1)

grouped_df5_1.rename(columns ={'Organ harvested':'Organ'},inplace=True)

grouped_df5_1['Organ'] = 2

grouped_df5_1["Date"]=grouped_df5_1.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45, "12_March":48,"13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })

grouped_df5_1 = grouped_df5_1.sort_values(by=['Date','EC_limit'])

grouped_df5_1['Date'] = grouped_df5_1['Date'].astype(float)
grouped_df5_1['Organ'] = grouped_df5_1['Organ'].astype(float)
grouped_df5_1['EC_limit'] = grouped_df5_1['EC_limit'].astype(float)

baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df5_1['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df5_1['Date'] - 1, unit='D')

grouped_df5_1 = update_df_values(grouped_df5_1, 1, 92)
grouped_df5_1 = update_df_values(grouped_df5_1, 93, 106)
grouped_df5_1 = update_df_values(grouped_df5_1, 107, 118)
grouped_df5_1 = update_df_values(grouped_df5_1, 119, 131)
grouped_df5_1 = update_df_values(grouped_df5_1, 132, 145)
grouped_df5_1 = update_df_values(grouped_df5_1, 146, 171)
grouped_df5_1 = update_df_values2(grouped_df5_1, 1, 106)
grouped_df5_1 = update_df_values2(grouped_df5_1, 107, 140)
grouped_df5_1= update_df_values2(grouped_df5_1, 141, 171)
grouped_df5_1 = update_df_values3(grouped_df5_1, 1, 118)
grouped_df5_1 = update_df_values3(grouped_df5_1, 119, 171)


grouped_df5_1


data5_2=pd.read_excel(r"C:\Users\Gizem\Desktop\IC/5 - Fruit cations-2.xlsx")

grouped_df5_2=data5_2.copy()

grouped_df5_2.rename(columns ={'EC limit':'EC_limit'}, inplace=True)

grouped_df5_2= grouped_df5_2.drop('Replication', axis=1)

grouped_df5_2.rename(columns ={'Organ harvested':'Organ'},inplace=True)

grouped_df5_2['Organ'] = 2

grouped_df5_2["Date"]=grouped_df5_2.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45, "12_March":48,"13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "3_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "9_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })

grouped_df5_2 = grouped_df5_2.sort_values(by=['Date','EC_limit'])

grouped_df5_2['Date'] = grouped_df5_2['Date'].astype(float)
grouped_df5_2['Organ'] = grouped_df5_2['Organ'].astype(float)
grouped_df5_2['EC_limit'] = grouped_df5_2['EC_limit'].astype(float)

baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df5_2['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df5_2['Date'] - 1, unit='D')

grouped_df5_2 = update_df_values(grouped_df5_2, 1, 92)
grouped_df5_2 = update_df_values(grouped_df5_2, 93, 106)
grouped_df5_2 = update_df_values(grouped_df5_2, 107, 118)
grouped_df5_2 = update_df_values(grouped_df5_2, 119, 131)
grouped_df5_2 = update_df_values(grouped_df5_2, 132, 145)
grouped_df5_2 = update_df_values(grouped_df5_2, 146, 171)
grouped_df5_2 = update_df_values2(grouped_df5_2, 1, 106)
grouped_df5_2 = update_df_values2(grouped_df5_2, 107, 140)
grouped_df5_2= update_df_values2(grouped_df5_2, 141, 171)
grouped_df5_2 = update_df_values3(grouped_df5_2, 1, 118)
grouped_df5_2 = update_df_values3(grouped_df5_2, 119, 171)

grouped_df5_2

data6_1=pd.read_excel(r"C:\Users\Gizem\Desktop\IC/6 - Roots cations-1.xlsx")

grouped_df6_1=data6_1.copy()

grouped_df6_1.rename(columns ={'EC limit':'EC_limit'}, inplace=True)

grouped_df6_1= grouped_df6_1.drop('Replication', axis=1)

grouped_df6_1.rename(columns ={'Organ harvested':'Organ'},inplace=True)
grouped_df6_1['Organ'] = 5

grouped_df6_1["Date"]=grouped_df6_1.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45,"12_March":48, "13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })

grouped_df6_1 = grouped_df6_1.sort_values(by=['Date','EC_limit'])

grouped_df6_1['Date'] = grouped_df6_1['Date'].astype(float)
grouped_df6_1['Organ'] = grouped_df6_1['Organ'].astype(float)
grouped_df6_1['EC_limit'] = grouped_df6_1['EC_limit'].astype(float)

baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df6_1['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df6_1['Date'] - 1, unit='D')

grouped_df6_1 = update_df_values(grouped_df6_1, 1, 92)
grouped_df6_1 = update_df_values(grouped_df6_1, 93, 106)
grouped_df6_1 = update_df_values(grouped_df6_1, 107, 118)
grouped_df6_1 = update_df_values(grouped_df6_1, 119, 131)
grouped_df6_1 = update_df_values(grouped_df6_1, 132, 145)
grouped_df6_1 = update_df_values(grouped_df6_1, 146, 171)
grouped_df6_1 = update_df_values2(grouped_df6_1, 1, 106)
grouped_df6_1 = update_df_values2(grouped_df6_1, 107, 140)
grouped_df6_1= update_df_values2(grouped_df6_1, 141, 171)
grouped_df6_1 = update_df_values3(grouped_df6_1, 1, 118)
grouped_df6_1 = update_df_values3(grouped_df6_1, 119, 171)

data7_1a=pd.read_excel(r"C:\Users\Gizem\Desktop\IC/7-total nitrogen-1a.xlsx")

grouped_df7_1a=data7_1a.copy()

grouped_df7_1a= grouped_df7_1a.drop('Replication', axis=1)


grouped_df7_1a['Organ'] = 3

grouped_df7_1a["Date"]=grouped_df7_1a.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45, "12_March":48,"13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })

grouped_df7_1a = grouped_df7_1a.sort_values(by=['Date','EC_limit'])

grouped_df7_1a['Date'] = grouped_df7_1a['Date'].astype(float)
grouped_df7_1a['Organ'] = grouped_df7_1a['Organ'].astype(float)
grouped_df7_1a['EC_limit'] = grouped_df7_1a['EC_limit'].astype(float)

baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df7_1a['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df7_1a['Date'] - 1, unit='D')


grouped_df7_1a = update_df_values(grouped_df7_1a, 1, 92)
grouped_df7_1a = update_df_values(grouped_df7_1a, 93, 106)
grouped_df7_1a = update_df_values(grouped_df7_1a, 107, 118)
grouped_df7_1a = update_df_values(grouped_df7_1a, 119, 131)
grouped_df7_1a = update_df_values(grouped_df7_1a, 132, 145)
grouped_df7_1a = update_df_values(grouped_df7_1a, 146, 171)
grouped_df7_1a = update_df_values2(grouped_df7_1a, 1, 106)
grouped_df7_1a = update_df_values2(grouped_df7_1a, 107, 140)
grouped_df7_1a = update_df_values2(grouped_df7_1a, 141, 171)
grouped_df7_1a = update_df_values3(grouped_df7_1a, 1, 118)
grouped_df7_1a = update_df_values3(grouped_df7_1a, 119, 171)

data7_1b=pd.read_excel(r"C:\Users\Gizem\Desktop\IC/7-total nitrogen-1b.xlsx")

grouped_df7_1b=data7_1b.copy()

grouped_df7_1b= grouped_df7_1b.drop('Replication', axis=1)

grouped_df7_1b['Organ'] = 4

grouped_df7_1b["Date"]=grouped_df7_1b.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45, "12_March":48,"13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })

grouped_df7_1b = grouped_df7_1b.sort_values(by=['Date','EC_limit'])

grouped_df7_1b['Date'] = grouped_df7_1b['Date'].astype(float)
grouped_df7_1b['Organ'] = grouped_df7_1b['Organ'].astype(float)
grouped_df7_1b['EC_limit'] = grouped_df7_1b['EC_limit'].astype(float)

baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df7_1b['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df7_1b['Date'] - 1, unit='D')

grouped_df7_1b = update_df_values(grouped_df7_1b, 1, 92)
grouped_df7_1b = update_df_values(grouped_df7_1b, 93, 106)
grouped_df7_1b = update_df_values(grouped_df7_1b, 107, 118)
grouped_df7_1b = update_df_values(grouped_df7_1b, 119, 131)
grouped_df7_1b = update_df_values(grouped_df7_1b, 132, 145)
grouped_df7_1b = update_df_values(grouped_df7_1b, 146, 171)
grouped_df7_1b = update_df_values2(grouped_df7_1b, 1, 106)
grouped_df7_1b = update_df_values2(grouped_df7_1b, 107, 140)
grouped_df7_1b = update_df_values2(grouped_df7_1b, 141, 171)
grouped_df7_1b = update_df_values3(grouped_df7_1b, 1, 118)
grouped_df7_1b = update_df_values3(grouped_df7_1b, 119, 171)

data7_1c=pd.read_excel(r"C:\Users\Gizem\Desktop\IC/7-total nitrogen-1c.xlsx")

grouped_df7_1c=data7_1c.copy()

grouped_df7_1c= grouped_df7_1c.drop('Replication', axis=1)

grouped_df7_1c['Organ'] = 5

grouped_df7_1c["Date"]=grouped_df7_1c.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45, "12_March":48,"13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })

grouped_df7_1c = grouped_df7_1c.sort_values(by=['Date','EC_limit'])

grouped_df7_1c['Date'] = grouped_df7_1c['Date'].astype(float)
grouped_df7_1c['Organ'] = grouped_df7_1c['Organ'].astype(float)
grouped_df7_1c['EC_limit'] = grouped_df7_1c['EC_limit'].astype(float)

baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df7_1c['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df7_1c['Date'] - 1, unit='D')

grouped_df7_1c = update_df_values(grouped_df7_1c, 1, 92)
grouped_df7_1c = update_df_values(grouped_df7_1c, 93, 106)
grouped_df7_1c = update_df_values(grouped_df7_1c, 107, 118)
grouped_df7_1c = update_df_values(grouped_df7_1c, 119, 131)
grouped_df7_1c = update_df_values(grouped_df7_1c, 132, 145)
grouped_df7_1c = update_df_values(grouped_df7_1c, 146, 171)
grouped_df7_1c = update_df_values2(grouped_df7_1c, 1, 106)
grouped_df7_1c = update_df_values2(grouped_df7_1c, 107, 140)
grouped_df7_1c = update_df_values2(grouped_df7_1c, 141, 171)
grouped_df7_1c = update_df_values3(grouped_df7_1c, 1, 118)
grouped_df7_1c = update_df_values3(grouped_df7_1c, 119, 171)

data7_1d=pd.read_excel(r"C:\Users\Gizem\Desktop\IC/7-total nitrogen-1d.xlsx")
grouped_df7_1d=data7_1d.copy()

grouped_df7_1d['Organ'] = 2

grouped_df7_1d["Date"]=grouped_df7_1d.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45, "12_March":48,"13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })

grouped_df7_1d = grouped_df7_1d.sort_values(by=['Date','EC_limit'])

grouped_df7_1d['Date'] = grouped_df7_1d['Date'].astype(float)
grouped_df7_1d['Organ'] = grouped_df7_1d['Organ'].astype(float)
grouped_df7_1d['EC_limit'] = grouped_df7_1d['EC_limit'].astype(float)


baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df7_1d['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df7_1d['Date'] - 1, unit='D')

grouped_df7_1d = update_df_values(grouped_df7_1d, 1, 92)
grouped_df7_1d = update_df_values(grouped_df7_1d, 93, 106)
grouped_df7_1d = update_df_values(grouped_df7_1d, 107, 118)
grouped_df7_1d = update_df_values(grouped_df7_1d, 119, 131)
grouped_df7_1d = update_df_values(grouped_df7_1d, 132, 145)
grouped_df7_1d = update_df_values(grouped_df7_1d, 146, 171)
grouped_df7_1d = update_df_values2(grouped_df7_1d, 1, 106)
grouped_df7_1d = update_df_values2(grouped_df7_1d, 107, 140)
grouped_df7_1d = update_df_values2(grouped_df7_1d, 141, 171)
grouped_df7_1d = update_df_values3(grouped_df7_1d, 1, 118)
grouped_df7_1d = update_df_values3(grouped_df7_1d, 119, 171)

data7_2=pd.read_excel(r"C:\Users\Gizem\Desktop\IC/7- total nitrogen-2.xlsx")

grouped_df7_2=data7_2.copy()

grouped_df7_2.rename(columns ={'EC limit':'EC_limit'}, inplace=True)

grouped_df7_2= grouped_df7_2.drop('Replication', axis=1)

grouped_df7_2.rename(columns ={'Organ harvested':'Organ'},inplace=True)

grouped_df7_2.rename(columns ={'N (%)':'N_percnt'},inplace=True)

grouped_df7_2['Organ'] = 3

grouped_df7_2["Date"]=grouped_df7_2.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45, "12_March":48,"13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })

grouped_df7_2 = grouped_df7_2.sort_values(by=['Date','EC_limit'])

grouped_df7_2['Date'] = grouped_df7_2['Date'].astype(float)
grouped_df7_2['Organ'] = grouped_df7_2['Organ'].astype(float)
grouped_df7_2['EC_limit'] = grouped_df7_2['EC_limit'].astype(float)
baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df7_2['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df7_2['Date'] - 1, unit='D')

grouped_df7_2 = update_df_values(grouped_df7_2, 1, 92)
grouped_df7_2 = update_df_values(grouped_df7_2, 93, 106)
grouped_df7_2 = update_df_values(grouped_df7_2, 107, 118)
grouped_df7_2 = update_df_values(grouped_df7_2, 119, 131)
grouped_df7_2 = update_df_values(grouped_df7_2, 132, 145)
grouped_df7_2 = update_df_values(grouped_df7_2, 146, 171)
grouped_df7_2 = update_df_values2(grouped_df7_2, 1, 106)
grouped_df7_2 = update_df_values2(grouped_df7_2, 107, 140)
grouped_df7_2= update_df_values2(grouped_df7_2, 141, 171)
grouped_df7_2 = update_df_values3(grouped_df7_2, 1, 118)
grouped_df7_2 = update_df_values3(grouped_df7_2, 119, 171)

data7_3=pd.read_excel(r"C:\Users\Gizem\Desktop\IC/7-total nitrogen-3.xlsx")

grouped_df7_3=data7_3.copy()
grouped_df7_3.rename(columns ={'EC limit':'EC_limit'}, inplace=True)
grouped_df7_3= grouped_df7_3.drop('Replication', axis=1)

grouped_df7_3.rename(columns ={'Organ harvested':'Organ'},inplace=True)
grouped_df7_3.rename(columns ={'N (%)':'N_percnt'},inplace=True)
grouped_df7_3['Organ'] = 2

grouped_df7_3["Date"]=grouped_df7_3.Date.map({"25_January":1, "02_February":9, "06_February":13, "09_February":16, "16_February":23, "21_February":28, "22_February":29, "23_February":30, "27_February":34, 
                   "02_March":38, "08_March":44, "09_March":45,"12_March":48, "13_March":49, "14_March":50, "16_March":52, "19_March":55, "21_March":57, "23_March":59, "25_March":61, "26_March":62, "27_March":63, "28_March":64, "30_March":66, 
                   "02_April":69, "03_April":70, "04_April":71, "05_April":72, "06_April":73, "09_April":76, "10_April":77, "11_April":78, "13_April":80, "15_April":82, "17_April":84, "19_April":86, "20_April":87, "22_April":89, "23_April":90, "24_April":91, "25_April":92, "26_April":93, "27_April":94, "29_April":96, "30_April":97, 
                   "01_May":98, "02_May":99, "03_May":100, "04_May":101, "05_May":102, "06_May":103, "08_May":105, "09_May":106, "10_May":107, "11_May":108, "12_May":109, "13_May":110, "14_May":111, "15_May":112, "16_May":113, "17_May":114, "18_May":115, "20_May":117, "21_May":118, "22_May":119, "24_May":121, "27_May":124, "28_May":125, "29_May":126, "30_May":127, "31_May":128, 
                   "01_June":129, "02_June":130, "03_June":131, "04_June":132, "05_June":133, "06_June":134, "07_June":135, "08_June":136, "09_June":137, "10_June":138, "11_June":139, "12_June":140, "13_June":141, "14_June":142, "15_June":143, "17_June":145, "18_June":146, "19_June":147, "20_June":148, "21_June":149, "22_June":150, "23_June":151, "24_June":152, "25_June":153, "27_June":155, "28_June":156, "29_June":157, 
                   "01_July":159, "03_July":161, "04_July":162, "05_July":163, "06_July":164, "07_July":165, "08_July":166, "09_July":167, "10_July":168, "11_July":169, "12_July":170, "13_July":171 })

grouped_df7_3 = grouped_df7_3.sort_values(by=['Date','EC_limit'])

grouped_df7_3['Date'] = grouped_df7_3['Date'].astype(float)
grouped_df7_3['Organ'] = grouped_df7_3['Organ'].astype(float)
grouped_df7_3['EC_limit'] = grouped_df7_3['EC_limit'].astype(float)

baslangic_tarihi = '2024-01-25'

# 'Date' sütunundaki gün sayılarını başlangıç tarihine ekleyerek yeni bir 'datetime' sütunu oluşturma
grouped_df7_3['datetime'] = pd.to_datetime(baslangic_tarihi) + pd.to_timedelta(grouped_df7_3['Date'] - 1, unit='D')

grouped_df7_3 = update_df_values(grouped_df7_3, 1, 92)
grouped_df7_3 = update_df_values(grouped_df7_3, 93, 106)
grouped_df7_3 = update_df_values(grouped_df7_3, 107, 118)
grouped_df7_3 = update_df_values(grouped_df7_3, 119, 131)
grouped_df7_3 = update_df_values(grouped_df7_3, 132, 145)
grouped_df7_3 = update_df_values(grouped_df7_3, 146, 171)
grouped_df7_3 = update_df_values2(grouped_df7_3, 1, 106)
grouped_df7_3 = update_df_values2(grouped_df7_3, 107, 140)
grouped_df7_3= update_df_values2(grouped_df7_3, 141, 171)
grouped_df7_3 = update_df_values3(grouped_df7_3, 1, 118)
grouped_df7_3 = update_df_values3(grouped_df7_3, 119, 171)


birlesik2 =pd.concat([grouped_df1_1b, grouped_df1_1a,grouped_df1_1c,grouped_df1_1d], axis=0)

birlesik2 =pd.concat([ birlesik2,grouped_df1_2,grouped_df1_3,grouped_df2_1,grouped_df3_1,grouped_df3_2
                    ,grouped_df4_1,grouped_df5_1,grouped_df5_2,grouped_df6_1,grouped_df7_1a
                    ,grouped_df7_1b,grouped_df7_1c,grouped_df7_1d,grouped_df7_2,grouped_df7_3],axis=0)

clim=pd.read_excel(r"C:\Users\Gizem\Desktop\IC/climate_data.xlsx")
emis=pd.read_excel(r"C:\Users\Gizem\Desktop\IC/cseasonal_emission.xlsx")
clim1=clim.copy()
emis1=emis.copy()
# İlk iki DataFrame'i birleştir


emis1= emis1.drop('Block', axis=1)

emis1= emis1.drop('Cultivar', axis=1)

emis1= emis1.drop('Identifier', axis=1)
clim1= clim1.drop('Cultivar', axis=1)
birlesik2k = pd.merge(clim1, emis1, on=['Date'], how='outer')


birlesik2_f = birlesik2k[~birlesik2k['Date'].dt.month.isin([8, 9, 10, 11]) & (birlesik2k['Date'].dt.year != 2020)]
birlesik2_f = birlesik2_f.sort_values(by=['Date'])
birlesik2_f.rename(columns ={'Date':'datetime'}, inplace=True)
birlesik2_f['datetime'] = birlesik2_f['datetime'].apply(lambda x: x.replace(year=2024))


imp = IterativeImputer(RandomForestRegressor(), max_iter=10, random_state=0)

# Eğitim için sadece sayısal sütunları kullanın
numerical_data = birlesik2_f.select_dtypes(include=[np.number])

# Modeli eğit ve eksik verileri doldur
imputed_data = imp.fit_transform(numerical_data)

# İmpute edilmiş veriyi DataFrame'e geri dönüştür
gops_imputed = pd.DataFrame(imputed_data, columns=numerical_data.columns, index=numerical_data.index)

# İmpute edilmiş verileri asıl DataFrame ile birleştir
gops_filled = birlesik2_f.copy()
gops_filled[numerical_data.columns] = gops_imputed
birlesik2_f=gops_filled

birlesik2_f.rename(columns ={'EC':'EC_limit'}, inplace=True)

birlesik2 =pd.concat([ birlesik2,  birlesik2_f], axis=0)

birlesik2['NS_residual_perplant'] = birlesik2['NS_residual_perplant'].fillna(0)

birlesik2['NS_added_perplant'] = birlesik2['NS_added_perplant'].fillna(0)

birlesik2['NS_new_perplant'] = birlesik2['NS_new_perplant'].fillna(0)

birlesik2['Organ'] = birlesik2['Organ'].fillna(1)

start_date = '2024-02-20'
end_date = '2024-07-18'

# Belirli tarih aralığında 'truss' sütunundaki NaN değerleri rastgele doldurmak için mask oluşturun
mask = (birlesik2['datetime'] >= start_date) & (birlesik2['datetime'] <= end_date) & birlesik2['Trusses_number'].isna()

# mask.sum() yerine doğrudan mask kullanarak NaN değerleri için rastgele sayılar üretin
# Bu, maskelenen ve NaN olan satır sayısı ile tam olarak eşleşen rastgele sayılar üretir
random_values = np.random.randint(15, 17, size=mask.sum())

# Maskelenen ve NaN olan hücreleri rastgele değerlerle doldurun
birlesik2.loc[mask, 'Trusses_number'] = random_values
start_date = '2024-01-26'
end_date = '2024-02-19'

# Belirli tarih aralığında 'truss' sütunundaki NaN değerleri rastgele doldurmak için mask oluşturun
mask = (birlesik2['datetime'] >= start_date) & (birlesik2['datetime'] <= end_date) & birlesik2['Trusses_number'].isna()

# mask.sum() yerine doğrudan mask kullanarak NaN değerleri için rastgele sayılar üretin
# Bu, maskelenen ve NaN olan satır sayısı ile tam olarak eşleşen rastgele sayılar üretir
random_values = np.random.randint(10, 15, size=mask.sum())

# Maskelenen ve NaN olan hücreleri rastgele değerlerle doldurun
birlesik2.loc[mask, 'Trusses_number'] = random_values

specific_date = '2024-01-25'


# Belirli tarih aralığında 'truss' sütunundaki NaN değerleri rastgele doldurmak için mask oluşturun
birlesik2.loc[(birlesik2['datetime'] == specific_date) & (birlesik2['Trusses_number'].isna()), 'Trusses_number'] = 0

birlesik2= birlesik2.drop('Dry_matter_pertruss', axis=1)

birlesik2= birlesik2.drop('Date', axis=1)

birlesik2['Dry_matter_perplant'] = birlesik2['Dry_matter_perplant'].fillna(method='ffill')  # Önce bir önceki değerle doldur
birlesik2['Dry_matter_perplant'] = birlesik2['Dry_matter_perplant'].fillna(method='bfill')  

birlesik2 =  birlesik2.sort_values(by=['Organ','Dry_matter_perplant'])

conditions = [
    (birlesik2['Substrate'] == 'Rock wool'),
    (birlesik2['Substrate'] == 'Coir, water-logged'),
    (birlesik2['Substrate'] == 'Coir'),
    (birlesik2['Substrate'] == 'Perlite/wood fiber')
]

choices = [1, 2, 3,4]

birlesik2['Substrate'] = np.select(conditions, choices)

birlesik2['Substrate'].fillna(pd.Series(np.random.choice([1.0, 2.0, 3.0, 4.0], size=len(birlesik2))), inplace=True)

birlesik2= birlesik2.drop('Dry matter (g)', axis=1)

imp = IterativeImputer(RandomForestRegressor(), max_iter=10, random_state=0)

# Eğitim için sadece sayısal sütunları kullanın
numerical_data = birlesik2.select_dtypes(include=[np.number])

# Modeli eğit ve eksik verileri doldur
imputed_data = imp.fit_transform(numerical_data)

# İmpute edilmiş veriyi DataFrame'e geri dönüştür
gops_imputed = pd.DataFrame(imputed_data, columns=numerical_data.columns, index=numerical_data.index)

# İmpute edilmiş verileri asıl DataFrame ile birleştir
gops_filled2 = birlesik2.copy()
gops_filled2[numerical_data.columns] = gops_imputed

birlesik2=gops_filled2

t_min, t_max = birlesik2['T_avg'].min(), birlesik2['T_avg'].max()

# 'water' değerlerini 't_avg'ye bağlı olarak 1.2 ile 1.6 arasında atayın
birlesik2['Water'] = birlesik2['T_avg'].apply(lambda x: 1.2 + (1.6 - 1.2) * ((x - t_min) / (t_max - t_min)))

birlesik2c=birlesik2.copy()


planting_date = pd.Timestamp('2024-01-25')
birlesik2c['days_since_planting'] = (birlesik2c['datetime'] - planting_date).dt.days

# Büyüme evrelerini tanımlayalım ve her bir satır için büyüme evresini belirleyelim
def determine_growth_stage(days):
    if days <= 10:
        return 'Çimlenme'
    elif days <= 52:
        return 'Fidan Gelişim'
    elif days <= 112:
        return 'Çiçeklenme'
    elif days <= 182:
        return 'Meyve Gelişim'
    else:
        return 'Olgunlaşma'

birlesik2c['growth_stage'] = birlesik2c['days_since_planting'].apply(determine_growth_stage)

# Girdi ve hedef değişkenlerimizi belirleyelim
input_features = ['days_since_planting', 'T_avg', 'CO2_avg' ]
target_variables = ['Na', 'K', 'Mg', 'Ca', 'N_percnt', 'Water','EC_limit']

# Girdi ve hedef değişkenlerin ilk birkaç satırını gösterelim
birlesik2c[input_features + target_variables]

y_ec = birlesik2c['EC_limit']
X = birlesik2c[['days_since_planting', 'T_avg', 'CO2_avg']]
X_transformed_ec = X.copy()

# Sıcaklık (T_avg) ve CO2_avg için log dönüşümü
X_transformed_ec['T_avg_log'] = np.log(X_transformed_ec['T_avg'] + 1)
X_transformed_ec['CO2_avg_log'] = np.log(X_transformed_ec['CO2_avg'] + 1)

# 'days_since_planting' için polinom özellikler (kare)
X_transformed_ec['days_since_planting_squared'] = X_transformed_ec['days_since_planting'] ** 2

# Orijinal 'T_avg' ve 'CO2_avg' sütunlarını kaldıralım
X_transformed_ec.drop(['T_avg', 'CO2_avg'], axis=1, inplace=True)

# Yeniden model kurma ve eğitimi
X_train_ec_transformed, X_test_ec_transformed, y_train_ec, y_test_ec = train_test_split(X_transformed_ec, y_ec, test_size=0.2, random_state=42)

rf_model_ec_transformed = RandomForestRegressor(random_state=42)
rf_model_ec_transformed.fit(X_train_ec_transformed, y_train_ec)

# Test seti üzerinde tahminler yapmak
y_pred_ec_transformed = rf_model_ec_transformed.predict(X_test_ec_transformed)

# Modelin performansını değerlendirme
mse_ec_transformed = mean_squared_error(y_test_ec, y_pred_ec_transformed)
r2_ec_transformed = r2_score(y_test_ec, y_pred_ec_transformed)

mse_ec_transformed, r2_ec_transformed


# Besin konsantrasyonları için model kurma ve eğitimi
y_nutrients = birlesik2c[['Na', 'K', 'Mg', 'Ca', 'N_percnt']]  # Hedef değişkenler

# Özelliklerin daha önce dönüştürüldüğü veri setini kullanma
X_nutrients = X_transformed_ec

# Veri setini eğitim ve test seti olarak bölmek
X_train_nutrients, X_test_nutrients, y_train_nutrients, y_test_nutrients = train_test_split(X_nutrients, y_nutrients, test_size=0.2, random_state=42)

# Çoklu çıktı regresyon modelini eğitmek
rf_model_nutrients = MultiOutputRegressor(RandomForestRegressor(random_state=42))
rf_model_nutrients.fit(X_train_nutrients, y_train_nutrients)

# Test seti üzerinde tahminler yapmak
y_pred_nutrients = rf_model_nutrients.predict(X_test_nutrients)

# Modelin performansını değerlendirme
mse_nutrients = mean_squared_error(y_test_nutrients, y_pred_nutrients, multioutput='raw_values')
r2_nutrients = r2_score(y_test_nutrients, y_pred_nutrients, multioutput='raw_values')

mse_nutrients, r2_nutrients


# Dikim tarihinden itibaren geçen gün sayısını hesaplama
planting_date = pd.Timestamp('2024-01-25')
birlesik2c['days_since_planting'] = (birlesik2c['datetime'] - planting_date).dt.days

# Özellik dönüşümleri uygulama
birlesik2c['T_avg_log'] = np.log(birlesik2c['T_avg'] + 1)
birlesik2c['CO2_avg_log'] = np.log(birlesik2c['CO2_avg'] + 1)
birlesik2c['days_since_planting_squared'] = birlesik2c['days_since_planting'] ** 2

# Water için girdi ve hedef değişkenler
X_new = birlesik2c[['days_since_planting', 'T_avg_log', 'CO2_avg_log', 'days_since_planting_squared']]
y_new_water = birlesik2c['Water']

# Veri setini eğitim ve test seti olarak bölmek
X_train_new, X_test_new, y_train_new_water, y_test_new_water = train_test_split(X_new, y_new_water, test_size=0.2, random_state=42)

# Rastgele Orman modelini eğitmek
rf_model_new_water = RandomForestRegressor(random_state=42)
rf_model_new_water.fit(X_train_new, y_train_new_water)

# Performans değerlendirme
y_pred_new_water = rf_model_new_water.predict(X_test_new)
mse_new_water = mean_squared_error(y_test_new_water, y_pred_new_water)
r2_new_water = r2_score(y_test_new_water, y_pred_new_water)

mse_new_water, r2_new_water


# Gerekli sütunları doğru bir şekilde oluşturalım
birlesik2c['days_since_planting'] = (pd.to_datetime(birlesik2c['datetime']) - pd.Timestamp('2024-01-25')).dt.days
birlesik2c['T_avg_log'] = np.log(birlesik2c['T_avg'] + 1)
birlesik2c['CO2_avg_log'] = np.log(birlesik2c['CO2_avg'] + 1)

# N2O emisyonu için girdi ve hedef değişkenler (düzeltme yapıldı)
X_emissions = birlesik2c[['days_since_planting', 'T_avg_log', 'CO2_avg_log', 'EC_limit', 'Na', 'K', 'Mg', 'Ca', 'N_percnt']]
y_n2o = birlesik2c['N2O_f1']  # N2O emisyonu hedef değişken

# Veri setini eğitim ve test seti olarak bölmek
X_train_n2o, X_test_n2o, y_train_n2o, y_test_n2o = train_test_split(X_emissions, y_n2o, test_size=0.2, random_state=42)

# N2O için Rastgele Orman Regresyon modelini eğitmek
rf_model_n2o = RandomForestRegressor(random_state=42)
rf_model_n2o.fit(X_train_n2o, y_train_n2o)

# Test seti üzerinde tahminler yapmak ve performans değerlendirme
y_pred_n2o = rf_model_n2o.predict(X_test_n2o)
mse_n2o = mean_squared_error(y_test_n2o, y_pred_n2o)
r2_n2o = r2_score(y_test_n2o, y_pred_n2o)

mse_n2o, r2_n2o



y_co2 = birlesik2c['CO2_f1']  # CO2 emisyonu hedef değişken

# Veri setini eğitim ve test seti olarak bölmek
X_train_co2, X_test_co2, y_train_co2, y_test_co2 = train_test_split(X_emissions, y_co2, test_size=0.2, random_state=42)

# CO2 için Rastgele Orman Regresyon modelini eğitmek
rf_model_co2 = RandomForestRegressor(random_state=42)
rf_model_co2.fit(X_train_co2, y_train_co2)

# Test seti üzerinde tahminler yapmak ve performans değerlendirme
y_pred_co2 = rf_model_co2.predict(X_test_co2)
mse_co2 = mean_squared_error(y_test_co2, y_pred_co2)
r2_co2 = r2_score(y_test_co2, y_pred_co2)

mse_co2, r2_co2


param_dist_co2 = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Randomized Search CV kullanarak hiperparametre ayarlama
random_search_co2 = RandomizedSearchCV(estimator=rf_model_co2, param_distributions=param_dist_co2, n_iter=10, cv=3, 
                                       scoring='r2', n_jobs=-1, random_state=42, verbose=2)
random_search_co2.fit(X_train_co2, y_train_co2)

# En iyi parametreleri ve bu parametrelerle elde edilen en iyi skoru göster
best_params_co2 = random_search_co2.best_params_
best_score_co2 = random_search_co2.best_score_

best_params_co2, best_score_co2


# CH4 emisyonu için girdi ve hedef değişkenler
y_ch4 = birlesik2c['CH4_f1']  # CH4 emisyonu hedef değişken

# Veri setini eğitim ve test seti olarak bölmek
X_train_ch4, X_test_ch4, y_train_ch4, y_test_ch4 = train_test_split(X_emissions, y_ch4, test_size=0.2, random_state=42)

# CH4 için Rastgele Orman Regresyon modelini eğitmek
rf_model_ch4 = RandomForestRegressor(random_state=42)
rf_model_ch4.fit(X_train_ch4, y_train_ch4)

# Test seti üzerinde tahminler yapmak ve performans değerlendirme
y_pred_ch4 = rf_model_ch4.predict(X_test_ch4)
mse_ch4 = mean_squared_error(y_test_ch4, y_pred_ch4)
r2_ch4 = r2_score(y_test_ch4, y_pred_ch4)

mse_ch4, r2_ch4


param_dist_ch4_reduced = {
    'n_estimators': [100, 200],  # Daha az ağaç sayısı seçeneği
    'max_depth': [None, 20],  # Derinlik için daha az seçenek
    'min_samples_split': [2, 5],  # Minimum örnek bölme için daha az seçenek
    'min_samples_leaf': [1, 2]  # Yaprak düğümünde minimum örnek sayısı için daha az seçenek
}

# Randomized Search CV kullanarak CH4 için hiperparametre ayarlama (daha az parametre ile)
random_search_ch4_reduced = RandomizedSearchCV(estimator=rf_model_ch4, param_distributions=param_dist_ch4_reduced, n_iter=5, cv=3, 
                                               scoring='r2', n_jobs=-1, random_state=42, verbose=2)
random_search_ch4_reduced.fit(X_train_ch4, y_train_ch4)

# En iyi parametreleri ve bu parametrelerle elde edilen en iyi skoru göster
best_params_ch4_reduced = random_search_ch4_reduced.best_params_
best_score_ch4_reduced = random_search_ch4_reduced.best_score_

best_params_ch4_reduced, best_score_ch4_reduced


X = birlesik2c[['T_avg_log', 'CO2_avg_log', 'EC_limit', 'Na', 'K', 'Mg', 'Ca', 'N_percnt']]
y = birlesik2c['Substrate']  # Bu değişken gerçekte veri setinde olmalıdır

# Veri setini eğitim ve test seti olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rastgele Orman Sınıflandırıcısını kurma ve eğitme
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma ve performans değerlendirme
y_pred = clf.predict(X_test)

print("Doğruluk Skoru:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))

