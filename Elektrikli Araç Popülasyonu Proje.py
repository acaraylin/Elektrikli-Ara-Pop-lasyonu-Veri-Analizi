# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import Lasso
from plotly.offline import plot
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

data=pd.read_csv('Electric_Vehicle_Population_Data.csv')
df = pd.DataFrame(data)
df.shape
df.head()

plt.figure(figsize=(12, 6))
df['Make'].value_counts().plot(kind='bar')
plt.xlabel('Marka')
plt.ylabel('Frekans')
plt.title('Araç Markalarına Göre Frekans Dağılımı')
plt.xticks(rotation=90)
plt.show()

# Eksik Veri Kontrol Etme
missing_values = df.isnull().sum()
print("Eksik veri sayısı her sütun için:")
print(missing_values)

# Eksik veri sayılarını bir veri çerçevesine dönüştürme
missing_values_df = pd.DataFrame({'Sütun': missing_values.index, 'Eksik Veri Sayısı': missing_values.values})

# Excel dosyasına yazdırma
missing_values_df.to_excel("eksik_veri_sayilari.xlsx", index=False)

# Gereksiz sütunları Silme 
df.drop('Legislative District', axis=1, inplace=True)
df.drop(['VIN (1-10)'],axis=1,inplace=True)
missing_values = df.isnull().sum()

# Eksik Verileri İçeren Sütunları Kontrol Etme
columns_with_missing_values = df.columns[df.isnull().any()]

# Eksik Verileri Doldurma
for column in columns_with_missing_values:
    if df[column].dtype == 'object':  
        df[column].fillna(df[column].mode()[0], inplace=True)
    else: 
        df[column].fillna(df[column].median(), inplace=True)
        
# Eksik veri kontrol etme ve Excel'e yazdırma
missing_values_after = df.isnull().sum()
missing_values_after_df = pd.DataFrame({'Sütun': missing_values_after.index, 'Eksik Veri Sayısı': missing_values_after.values})
missing_values_after_df.to_excel("eksik_veri_sayilari_sonra.xlsx", index=False)


#Sınıf Etiketi 
print(df['Electric Vehicle Type'].unique())
bev_data = df[df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)']
phev_data = df[df['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle (PHEV)']
print("BEV veri sayısı:", len(bev_data))
print("PHEV veri sayısı:", len(phev_data))
class_distribution = df['Electric Vehicle Type'].value_counts()
print(class_distribution)

# BEV ve PHEV araçlarını içeren veriyi oluşturma
bev_data = df[df['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)']
phev_data = df[df['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle (PHEV)']

# Araç markalarının BEV ve PHEV araç sayılarını hesaplama
bev_counts_per_make = bev_data['Make'].value_counts()
phev_counts_per_make = phev_data['Make'].value_counts()

# Grafik 
plt.figure(figsize=(12, 6))
plt.bar(bev_counts_per_make.index, bev_counts_per_make.values, color='orange', label='BEV')
plt.bar(phev_counts_per_make.index, phev_counts_per_make.values, color='lightgreen', label='PHEV')
plt.xlabel('Marka')
plt.ylabel('Araç Sayısı')
plt.title('Marka Bazında BEV ve PHEV Araç Sayısı')
plt.xticks(rotation=90)
plt.legend()
plt.show()

# 'Vehicle Location' sütununu parçalayarak 'langitude' ve 'longitude' sütunlarını oluşturma
df[['langitude', 'longitude']] = df['Vehicle Location'].str.extract(r'POINT \((-?\d+\.\d+) (-?\d+\.\d+)\)')

# Oluşturulan 'langitude' ve 'longitude' sütunlarının veri türünü float olarak dönüştürme
df['langitude'] = df['langitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)

# İlk 5 satırı ve oluşturulan sütunları gösterme
print(df[['langitude', 'longitude']].head())

# Konuma göre araç popülasyonunu gösterme
plt.figure(figsize=(10, 8))
sns.scatterplot(x='longitude', y='langitude', data=df)
plt.title('Konuma Göre Araç Popülasyonu')
plt.xlabel('Longitude')
plt.ylabel('Langitude')
plt.show()

# Belirli bir aralık dışındaki longitude ve langitude değerlerine sahip verileri çıkarmak için filtreleme
filtered_df = df[(df['longitude'] >= 25) & (df['longitude'] <= 55) & 
                 (df['langitude'] >= -140) & (df['langitude'] <= 0)]

# Belirtilen aralık dışındaki verileri çıkarmak için ters filtreyi uygula
filtered_df_outside_range = df[~((df['longitude'] >= 25) & (df['longitude'] <= 55) & 
                                 (df['langitude'] >= -140) & (df['langitude'] <= 0))]

# Belirtilen aralık dışındaki verileri veri setinden çıkar
cleaned_df = df.drop(filtered_df_outside_range.index)

# Yeni scatter plot grafiğini oluşturma
plt.figure(figsize=(10, 8))
sns.scatterplot(x='longitude', y='langitude', data=filtered_df)
plt.title('Aykırı Değerler Çıkarılmış Araç Popülasyonu')
plt.xlabel('Longitude')
plt.ylabel('Langitude')
plt.show()

# Şu anki yılı belirt
current_year = 2024

# Model yılını alarak aracın yaşını hesapla
df['Vehicle Age'] = current_year - df['Model Year']

# Oluşturulan araç yaşını göster
print(df[['Model Year', 'Vehicle Age']].head())

# Min-Max ölçekleyiciyi oluştur
scaler = MinMaxScaler()

# Vehicle Age sütununu tek boyutlu bir dizide al
vehicle_age_data = df['Vehicle Age'].values.reshape(-1, 1)

# Verileri ölçeklendirme
scaled_vehicle_age = scaler.fit_transform(vehicle_age_data)

# Ölçeklendirilmiş Vehicle Age değerlerini DataFrame'e dönüştürme
df['Scaled Vehicle Age'] = scaled_vehicle_age

# Ölçeklenmiş Vehicle Age değerlerinin histogramını çizme
plt.figure(figsize=(10, 6))
sns.histplot(df['Scaled Vehicle Age'], bins=20, kde=True)
plt.title('Ölçeklenmiş Araç Yaşı Dağılımı')
plt.xlabel('Ölçeklenmiş Araç Yaşı')
plt.ylabel('Frekans')
plt.show()

plt.figure(figsize=(12, 8))
sns.scatterplot(x='Electric Range', y='Scaled Vehicle Age', data=df)
plt.title('Ölçeklenmiş Araç Yaşı ve Menzili Dağılımı')
plt.xlabel('Elektrikli Araç Menzili')
plt.ylabel('Ölçeklenmiş Araç Yaşı')
plt.show()

# Her markanın araç sayısını hesaplama
car_counts_per_make = df['Make'].value_counts()

# Grafik 
plt.figure(figsize=(12, 6))
car_counts_per_make.plot(kind='bar')
plt.xlabel('Marka')
plt.ylabel('Araç Sayısı')
plt.title('Marka Bazında Araç Sayısı')
plt.xticks(rotation=90)  
plt.show()

# Tüm markaların araç sayısının ortalaması
average_car_count = df['Make'].value_counts().mean()

# Minimum eşik değeri 
threshold = average_car_count / 2

# Eşik değerinden fazla araca sahip olan markalar
above_threshold_makes = df['Make'].value_counts()[df['Make'].value_counts() > threshold]

# Sonuçlar
print("Ortalama araç sayısı:", average_car_count)
print("Minimum eşik değeri:", threshold)
print("Minimum eşik değerinden fazla araca sahip markalar:")
print(above_threshold_makes)

# Minimum araç sayısı eşiği
minimum_car_count = 2000  

# Eşiği geçmeyen markaları filtrele
selected_makes = car_counts_per_make[car_counts_per_make >= minimum_car_count].index

# Filtreleme
make_filter = df['Make'].isin(selected_makes)

# Eşiği geçen markaların veri seti
balanced_df = df[make_filter]

# Sonuçlar
print("Seçilen markaların araç sayıları:")
print(balanced_df['Make'].value_counts())
print("Yeni veri seti boyutu:", balanced_df.shape)

# Marka bazında araç sayısını hesapla
car_counts_per_make_filtered = balanced_df['Make'].value_counts()

# Çubuk grafik oluşturma
plt.figure(figsize=(12, 6))
sns.barplot(x=car_counts_per_make_filtered.index, y=car_counts_per_make_filtered.values)
plt.xlabel('Marka')
plt.ylabel('Araç Sayısı')
plt.title('Filtrelenmiş Veri Setinde Marka Bazında Araç Sayısı')
plt.xticks(rotation=90)
plt.show()

# Marka ve elektrikli araç tipine göre grupla ve ortalama menzili hesapla
mean_ranges_balanced = balanced_df.groupby(['Make', 'Electric Vehicle Type'])['Electric Range'].mean()

# Excel dosyasına yazdırma
mean_ranges_balanced.to_excel("ortalama_menzil_dengelenmis.xlsx", index=True)
print("Dengelenmiş veri seti için ortalama menzil verileri başarıyla Excel dosyasına yazdırıldı.")

# PCA modelini oluşturma
pca = PCA(n_components=2)  

# Bağımsız değişkenleri (X) al
X = df[['Postal Code', 'Model Year', 'Electric Range', 'Base MSRP', 'DOL Vehicle ID', 
        '2020 Census Tract', 'langitude', 'longitude', 'Vehicle Age', 'Scaled Vehicle Age']]

# PCA uygulama
X_pca = pca.fit_transform(X)

# Oluşturulan bileşenlerin boyutlarını görüntüleme
print("PCA'dan önce veri setinin boyutu:", X.shape)
print("PCA'dan sonra veri setinin boyutu:", X_pca.shape)

# Bağımsız değişkenler (X) ve bağımlı değişken (y) olarak veriyi ayırma
X = balanced_df[['Make', 'Vehicle Age', 'Electric Vehicle Type']]
y = balanced_df['Electric Range']

# Kategorik değişkenleri kodlamak için LabelEncoder kullanma
label_encoder = LabelEncoder()
X['Make'] = label_encoder.fit_transform(X['Make'])
X['Electric Vehicle Type'] = label_encoder.fit_transform(X['Electric Vehicle Type'])

# Eğitim ve test kümelerine ayırma (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ölçeklendirme işlemi
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelin eğitimi
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)

# Test verisi ile modeli değerlendirme
y_pred = knn_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Model Doğruluğu:", accuracy)


# K-means kümeleme modelini oluşturun ve uygulayın
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(filtered_df[['langitude', 'longitude']])

# Küme merkezlerini ve kümeleri alın
cluster_centers = kmeans.cluster_centers_
clusters = kmeans.labels_

# Kümeleme sonuçlarını veri setine ekleyin
filtered_df['Cluster'] = clusters

# Kümeleme sonuçlarını görselleştirin
plt.figure(figsize=(10, 8))
sns.scatterplot(x='longitude', y='langitude', hue='Cluster', data=filtered_df, palette='pastel')
plt.scatter(cluster_centers[:, 1], cluster_centers[:, 0], marker='x', s=200, c='red')
plt.title('K-means Kümeleme Sonuçları')
plt.xlabel('Longitude')
plt.ylabel('Langitude')
plt.show()

# Veri setini özellikler ve hedef değişken olarak ayırma
X = balanced_df[['Electric Range', 'Electric Vehicle Type']]
y = balanced_df['Make']

# Elektrikli araç tiplerini (BEV ve PHEV) sayma
bev_count = len(X[X['Electric Vehicle Type'] == 'Battery Electric Vehicle (BEV)'])
phev_count = len(X[X['Electric Vehicle Type'] == 'Plug-in Hybrid Electric Vehicle (PHEV)'])

# Toplam araç sayısını belirleme
total_count = len(X)

# BEV ve PHEV araçların oranını hesaplama
bev_ratio = bev_count / total_count
phev_ratio = phev_count / total_count

print("BEV araçların oranı:", bev_ratio)
print("PHEV araçların oranı:", phev_ratio)

# Veri setini özellikler ve hedef değişken olarak ayırma
X = balanced_df['Make']  # Özellikler: Marka
y = balanced_df['Electric Vehicle Type']  # Hedef değişken: Elektrikli araç tipi (BEV veya PHEV)

# Eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Markaları sayısal değerlere dönüştürme
label_encoder = LabelEncoder()
X_train_encoded = label_encoder.fit_transform(X_train)
X_test_encoded = label_encoder.transform(X_test)

# Bayes sınıflandırıcısını oluşturma ve eğitme
bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train_encoded.reshape(-1, 1), y_train)

# Test verilerini kullanarak tahmin yapma
y_pred = bayes_classifier.predict(X_test_encoded.reshape(-1, 1))

# Gelecekteki BEV ve PHEV araç popülasyonunun tahmini oranlarını hesaplama
bev_ratio = (y_pred == 'Battery Electric Vehicle (BEV)').sum() / len(y_pred)
phev_ratio = (y_pred == 'Plug-in Hybrid Electric Vehicle (PHEV)').sum() / len(y_pred)

print("Gelecekteki BEV araç popülasyonunun tahmini oranı:", bev_ratio)
print("Gelecekteki PHEV araç popülasyonunun tahmini oranı:", phev_ratio)

# Gelecekteki BEV ve PHEV araç popülasyonunun tahmini oranlarını listeye ekleme
populations = [bev_ratio, phev_ratio]

# Pasta grafiği oluşturma
labels = ['BEV', 'PHEV']
colors = ['orange', 'lightgreen']

plt.figure(figsize=(8, 6))
plt.pie(populations, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Gelecekteki BEV ve PHEV Araç Popülasyonu Tahmini Oranları')
plt.axis('equal')  
plt.show()

# Modelin doğruluğunu hesapla
accuracy = accuracy_score(y_test, y_pred)
print("Model Doğruluğu:", accuracy)

# Ortalama menzil sütunu oluşturma
balanced_df['Average Range'] = balanced_df.groupby('Make')['Electric Range'].transform('mean')

# Şarj istasyonu talebi sütunu oluşturma
balanced_df['Charge Station Demand'] = (balanced_df['Electric Range'] < balanced_df['Average Range']).astype(int)

# Özellikler ve hedef değişkeni seçme
X = balanced_df[['Electric Range', 'langitude', 'longitude']]
y = balanced_df['Charge Station Demand']

# Eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lojistik regresyon modeli oluşturma
logistic_model = LogisticRegression()

# Modeli eğitme
logistic_model.fit(X_train, y_train)

# Test verisi ile modeli değerlendirme
accuracy = logistic_model.score(X_test, y_test)
print("Lojistik Regresyon Modelinin Doğruluğu:", accuracy)

# Lojistik regresyon modeli kullanarak şarj istasyonu talebini tahmin etme
predicted_demand = logistic_model.predict(X_test)

# Tahmin edilen talebi veri çerçevesine ekleyerek görselleştirme
X_test_with_predictions = X_test.copy()
X_test_with_predictions['Predicted Demand'] = predicted_demand

# İlk 10 satırı gösterme
print(X_test_with_predictions.head(10))

# Tahmin edilen taleplerin histogramını çizme
plt.figure(figsize=(8, 6))
plt.hist(predicted_demand, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Şarj İstasyonu Talebi Tahmini Dağılımı')
plt.xlabel('Tahmini Talep')
plt.ylabel('Frekans')
plt.grid(True)
plt.show()

# Eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı modelini oluşturma
decision_tree_model = DecisionTreeClassifier(random_state=42)

# Modeli eğitme
decision_tree_model.fit(X_train, y_train)

# Test verisi ile modeli değerlendirme
y_pred = decision_tree_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Karar Ağacı Modelinin Doğruluğu:", accuracy)

# Karar ağacı modelinin performansını değerlendirme
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Confusion Matrix oluşturma
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Demand', 'Demand'], yticklabels=['No Demand', 'Demand'])
plt.title('Karar Ağacı Modeli İçin Confusion Matrix')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

# Eğitim ve test kümelerine ayırma (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest modelini oluşturma ve eğitme
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Test verisi ile modeli değerlendirme
y_pred = rf_model.predict(X_test)

# Tahmin olasılıklarını alma
y_pred_prob = rf_model.predict_proba(X_test)[:, 1]

# ROC eğrisini hesaplama
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# ROC eğrisini çizme
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Heatmap grafiğini oluşturma
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=['No Demand', 'Demand'], yticklabels=['No Demand', 'Demand'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of Random Forest Model')
plt.show()

# Özellikler ve hedef değişkeni seçme
X = balanced_df[['Electric Range', 'langitude', 'longitude']]
y = balanced_df['Charge Station Demand']

# Model seçimi ve parametre ayarlaması
models = {
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
    'LogisticRegression': LogisticRegression(),
    'GaussianNB': GaussianNB(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42)
}

# 10 kat çapraz geçerleme ile doğruluk ve AUC değerlerini hesaplama
results = {}
for name, model in models.items():
    cv_accuracy = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    cv_auc = cross_val_score(model, X, y, cv=10, scoring='roc_auc')
    results[name] = {
        'Accuracy': cv_accuracy.mean(),
        'AUC': cv_auc.mean()
    }

# Sonuçları görüntüleme
results_df = pd.DataFrame(results)
with pd.option_context('display.max_columns', None):
    print(results_df)


