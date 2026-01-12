<p align="center">
  <img src="hrimagine.png" width="100%">
</p>

# 📊 Employee Attrition Intelligence System





#  Employee Attrition Intelligence System

Bu proje, bir şirketteki çalışanların işten ayrılma (Attrition) nedenlerini analiz etmek ve makine öğrenmesi modelleri ile ayrılma riski olan çalışanları önceden tahmin etmek amacıyla geliştirilmiştir.

##  Projenin Öne Çıkan Özellikleri
* **Uçtan Uca Pipeline:** Veri temizlemeden (EDA), özellik mühendisliğine (Feature Engineering) ve model hiperparametre optimizasyonuna (GridSearchCV) kadar tüm süreçleri kapsar.
* **Feature Engineering:** İK metriklerine dayalı, modelin başarısını artıran **NEW_** ön ekli özgün değişkenler üretilmiştir.
* **Dürüst Modelleme:** "Target Leakage" (Hedef Sızıntısı) analizi yapılarak modelin kopya çekmesi engellenmiş, gerçek hayat performansına odaklanılmıştır.

##  Veri Mühendisliği ve Yeni Değişkenler
Sıradan bir analizden farklı olarak, aşağıdaki stratejik değişkenler üretilmiştir:
1. **NEW_Income_Per_Total_Year:** Toplam tecrübe yılına göre kazanılan maaş oranı.
2. **NEW_Total_Satisfaction:** İş, ortam ve ilişki memnuniyetinin ağırlıklı ortalaması.
3. **NEW_Career_Loyalty_Ratio:** Kariyerinin ne kadarını bu şirkette geçirdiği.

##  Önemli Bulgular (Insights)
* **Fazla Mesai:** Fazla mesai yapan çalışanların ayrılma oranı yapmayanlara göre **3 kat** daha fazladır.
* **Maaş Faktörü:** Tecrübesine oranla düşük kazanan çalışanların ayrılma riski daha yüksektir.

##  Model Performansı
* **Accuracy:** %88
* **Algoritma:** Random Forest (Tuned)

---
