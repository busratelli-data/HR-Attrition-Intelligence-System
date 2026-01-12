
#İş Problemi: Stratejik İşgücü Analitiği ve Çalışan Kaybı Tahmini
#Problem Tanımı: Nitelikli işgücünün korunması, bankacılık gibi insan kaynağının kritik olduğu sektörlerde operasyonel verimlilik ve
#maliyet yönetimi açısından hayati önem taşır. Bir çalışanın işten ayrılması (turnover); sadece yeni işe alım maliyetlerini değil,
#aynı zamanda kurumsal hafıza kaybını ve ekip içi motivasyon düşüşünü de beraberinde getirir.


#Hedef: Bu projede, çalışanların demografik bilgileri, iş tatmin anketleri ve performans metriklerini kullanarak;
#1)Hangi çalışanların işten ayrılma riskinin daha yüksek olduğunu makine öğrenmesi modelleri ile önceden tespit etmek (Tahmine Dayalı Analitik).
#2)İşten ayrılmaları tetikleyen ana faktörleri (Maaş, Fazla Mesai, Kariyer Yolu vb.)
#istatistiksel yöntemlerle ortaya koyarak İK yönetimine proaktif çözüm stratejileri sunmaktır.

#Veri Seti Hikayesi: Veri seti, IBM veri bilimcileri tarafından oluşturulmuş hayali ancak gerçek iş dünyası dinamiklerini yansıtan bir İK veri setidir.
#Toplam 1470 çalışan ve 35 farklı değişkenden oluşmaktadır.

#Değişkenlerin Özeti (1470 Gözlem | 35 Değişken):
#Attrition: İşten ayrılma durumu (Evet/Hayır) - Hedef Değişken.
#Demografik Veriler: Yaş, Cinsiyet, Medeni Durum, Eğitim Seviyesi ve Alanı.
#İş İlişkili Veriler: Departman, Görev Rolü, Mevcut Seviye, Günlük/Aylık Kazanç, Fazla Mesai.
#Memnuniyet ve Denge: İş Tatmini, Çalışma Ortamı Memnuniyeti, İş-Yaşam Dengesi, İlişki Memnuniyeti
#Kariyer ve Geçmiş: Şirketteki Yıl Sayısı, Toplam Deneyim Yılı, Son Terfiden Beri Geçen Süre, Mevcut Yöneticiyle Geçirilen Süre.

################################################
# DATA SOURCE & ACKNOWLEDGMENT
################################################
# Dataset: IBM HR Analytics Employee Attrition & Performance
# Source: Kaggle (https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
# Note: This is a fictional data set created by IBM data scientists.
################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings

warnings.simplefilter(action="ignore")

# Görüntü Ayarları
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################

# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)


#1)Genel Resim (General Picture)
# Veri Okuma

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head()
df.info()
df.describe(include="all")

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


#2) Değişkenlerin Yakalanması

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)



#3)Kategorik ve Sayısal Değişken Analizi

def cat_summary(dataframe, col_name, plot=False):
    print(f"--- {col_name} ---")
    summary = pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
    })
    print(summary)
    print("##########################################\n")

    if plot:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.title(f"Distribution of {col_name}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show(block=True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(f"--- {numerical_col} ---")
    print(dataframe[numerical_col].describe(quantiles).T)
    print("##########################################\n")

    if plot:
        plt.figure(figsize=(8, 5))
        dataframe[numerical_col].hist(bins=20, color='skyblue', edgecolor='black')
        plt.xlabel(numerical_col)
        plt.title(f"Histogram of {numerical_col}")
        plt.tight_layout()
        plt.show(block=True)



for col in cat_cols:
    cat_summary(df, col, plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)


#4)Hedef Değişken Analizi (Analysis of Target Variable)

def target_summary_with_cat(dataframe, target, categorical_col):
    print(f"##################### {categorical_col} #####################")
    # Attrition'ı analiz için geçici olarak 1 ve 0'a çeviriyoruz
    target_num = target + "_num"
    dataframe[target_num] = dataframe[target].apply(lambda x: 1 if x == "Yes" else 0)

    summary = pd.DataFrame({
        "TARGET_MEAN": dataframe.groupby(categorical_col)[target_num].mean(),
        "Count": dataframe.groupby(categorical_col)[target_num].count()
    }).sort_values(by="TARGET_MEAN", ascending=False)

    print(summary)
    print("\n")


# Tüm kategorik değişkenler için analizimi çalıştırıyorum
for col in cat_cols:
    target_summary_with_cat(df, "Attrition", col)





#5) Korelasyon Analizi (Analysis of Correlation)


def find_correlation(dataframe, numeric_cols, target="Attrition_num"):
    print("##################### Correlation Matrix #####################")

    # Target değişkenin dataframe içinde olup olmadığını kontrol edelim
    cols_to_corr = [col for col in numeric_cols if col in dataframe.columns]
    if target in dataframe.columns:
        cols_to_corr = cols_to_corr + [target]
        print(dataframe[cols_to_corr].corr()[target].sort_values(ascending=False))
    else:
        print(
            f"Uyarı: {target} sütunu veri setinde bulunamadı. Sadece sayısal değişkenlerin kendi aralarındaki korelasyonu basılıyor.")
        print(dataframe[cols_to_corr].corr())

    # Isı Haritası (Heatmap)
    plt.figure(figsize=[12, 10])
    sns.heatmap(dataframe[cols_to_corr].corr(), annot=True, fmt=".2f", cmap="RdBu")
    plt.title("Değişkenlerin Birbirleriyle ve Attrition ile Korelasyonu")
    plt.show(block=True)


find_correlation(df, num_cols, target="Attrition_num")



#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################

# 1. Outliers (Aykırı Değerler)
# 2. Missing Values (Eksik Değerler)
# 3. Feature Extraction (Özellik Çıkarımı)
# 4. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
# 5. Feature Scaling (Özellik Ölçeklendirme)



#1)Outliers (Aykırı Değerler)

import matplotlib.pyplot as plt
import seaborn as sns

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95): # HR verisi için 0.05-0.95 daha güvenli
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#  Uygulama
print("--- Aykırı Değer Kontrolü ---")
for col in num_cols:
    print(f"{col}: {check_outlier(df, col)}")

# Aykırı Değerleri Baskılama (Capping)
for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

print("\n--- Baskılama Sonrası Kontrol ---")
for col in num_cols:
    print(f"{col}: {check_outlier(df, col)}")



#############################################
# 2. Missing Values (Eksik Değerler)
#############################################

# 1. Güvenli Kopya
df1 = df.copy()
print("df1 kopyası başarıyla oluşturuldu.")


#Eksik Değer Tablosu
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (n_miss / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, ratio], axis=1, keys=["Missing Values", "Ratio (%)"])

    print("\n--- Missing Value Summary ---\n")
    if missing_df.empty:
        print("Veri setinde eksik değer bulunmuyor.")
    else:
        print(missing_df)

    if na_name:
        return na_columns


#Eksik Değer Doldurma
def fill_missing_values(dataframe, num_cols, cat_cols):
    print("\n--- Filling Missing Values (If Any) ---\n")

    # Sayısal değişkenler için median (Aykırı değerlere karşı dirençli)
    for col in num_cols:
        if dataframe[col].isnull().sum() > 0:
            dataframe[col] = dataframe[col].fillna(dataframe[col].median())
            print(f"{col}  median ile dolduruldu.")

    # Kategorik değişkenler için mode (En sık tekrar eden değer)
    for col in cat_cols:
        if dataframe[col].isnull().sum() > 0:
            dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])
            print(f"{col}  mode ile dolduruldu.")

    return dataframe


# --- ÇALIŞTIRMA ---

# 2.1 Analiz
print("Eksik değer analizi başlatılıyor...")
missing_values_table(df1)

# 2.2 Doldurma
#HR verisinde eksik olmasa bile bu fonksiyonu çalıştırmak pipeline güvenliği sağlar
df1 = fill_missing_values(df1, num_cols=num_cols, cat_cols=cat_cols)

print("\ndf1 temizlendi ve Feature Extraction aşamasına hazır!")


#3) Feature Extraction (Özellik Çıkarımı)

def feature_extraction(df):
    df = df.copy()
    print("--- Feature Extraction Started (HR Mode) ---")

    # 1. Kariyer ve Kıdem Değişkenleri
    # Toplam kariyerinin ne kadarını bu şirkette geçirdi? (Loyalty)
    df["NEW_Career_Loyalty_Ratio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)

    # Terfi hızı (Şirketteki yılına oranla kaç yıldır terfi almadı?)
    df["NEW_Promotion_Wait_Ratio"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)

    # 2. Maaş ve Verimlilik
    # Toplam çalışma yılına göre maaş (Tecrübe başına kazanç)
    df["NEW_Income_Per_Total_Year"] = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)

    # Maaş grubunu segmentlere ayırma (Senin PurchaseDelayGroup mantığın gibi)
    df["NEW_Income_Group"] = pd.cut(df["MonthlyIncome"],
                                    bins=[0, 3000, 7000, 15000, 30000],
                                    labels=["Low", "Medium", "High", "VeryHigh"])

    # 3. İş-Yaşam ve Memnuniyet Segmentleri
    # Fazla mesai ve Düşük İş-Yaşam Dengesi (Burnout Riski)
    df["NEW_Is_Burnout_Risk"] = ((df["OverTime"] == "Yes") & (df["WorkLifeBalance"] <= 2)).astype(int)

    # Memnuniyetlerin ortalaması
    df["NEW_Total_Satisfaction"] = (df["EnvironmentSatisfaction"] + df["JobSatisfaction"] + df[
        "RelationshipSatisfaction"]) / 3

    # 4. Demografik Gruplar (Senin AgeGroup mantığın)
    df["NEW_Age_Group"] = pd.cut(df["Age"],
                                 bins=[0, 25, 35, 50, 100],
                                 labels=["Junior", "Mid_Level", "Senior", "Expert"])

    print("--- Feature Extraction Completed Successfully ---")
    return df


# Uygulama
df2 = feature_extraction(df1)


#4)Encoding (One-Hot Encoding)

# 1. Yeniden değişkenleri yakalanması (Yeni eklenen featurelar dahil)
cat_cols, num_cols, cat_but_car = grab_col_names(df2)

# 2. Attrition'ı (Target) encoding listesinden çıkarılması
# bağımlı değişkeni ayrı tutma hedefindeyim burada
cat_cols = [col for col in cat_cols if "Attrition" not in col]


# 3. One-Hot Encoder Fonksiyonu
def one_hot_encoder(df, categorical_cols, drop_first=True):
    # drop_first=True dedim çünkü "Dummy Variable Trap" (kukla değişken tuzağı) engelledim böylece
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)

    #  bool -> int dönüşümü
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'bool':
            df_encoded[col] = df_encoded[col].astype(int)
    return df_encoded


# 4. Uygulama
df2_encoded = one_hot_encoder(df2, cat_cols, drop_first=True)

# 5. Hedef Değişkeni (Attrition) Manuel Kodu
df2_encoded["Attrition"] = df2_encoded["Attrition"].apply(lambda x: 1 if x == "Yes" else 0)

print("\n--- One-Hot Encoding Tamamlandı ---")
print(f"Yeni Veri Seti Boyutu: {df2_encoded.shape}")
print(df2_encoded.head())



#5)Feature Scaling (Özellik Ölçeklendirme)

from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Sayısal değişkenleri ve hedef değişkeni ayırma
cat_cols, num_cols, cat_but_car = grab_col_names(df2_encoded)

# Hedef değişkenimiz "Attrition" olduğu için onu ölçeklendirme dışında tuttum burda
target_col = "Attrition"
num_cols = [col for col in num_cols if col not in [target_col]]

# 2. Sayısal özellikleri ölçeklendirme fonksiyonu
def feature_scaling(dataframe, numerical_columns):
    scaler = StandardScaler()
    dataframe[numerical_columns] = scaler.fit_transform(dataframe[numerical_columns])
    return dataframe

# Uygulama
df2_final = feature_scaling(df2_encoded, num_cols)

print("Scaling işlemi başarıyla tamamlandı!")

# 3. Final Korelasyon Analizi
plt.figure(figsize=(25, 12))
# np.number kullanarak tüm sayısal tipleri kapsadım burda
sns.heatmap(df2_final.select_dtypes(include=[np.number]).corr(),
            annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap (After Pre-processing & Feature Engineering)")
plt.show(block=True)


#############################################
# 6. FINAL CHECK & NA HANDLING (SON KONTROL)
#############################################

# Feature Extraction sonrası oluşabilecek olası boşlukları kontrol etmek için;
if df2_final.isnull().values.any():
    print(f"Uyarı: İşlemler sonrası {df2_final.isnull().sum().sum()} adet boş değer oluştu, temizleniyor...")
    df2_final = df2_final.fillna(0)
    print("Boş değerler 0 ile dolduruldu.")
else:
    print("Harika! Veri setinde hiçbir boş değer kalmadı, modellemeye hazır.")

# Sonsuz (inf) değer kontrolü (Bölme işlemlerinden kaynaklı oluşabilir)
if np.isinf(df2_final.select_dtypes(include=[np.number])).values.any():
    print("Uyarı: Sonsuz (inf) değerler tespit edildi, düzenleniyor...")
    df2_final = df2_final.replace([np.inf, -np.inf], 0)

print(f"Final Veri Seti Boyutu: {df2_final.shape}")










# -----------------------------
# MODEL PIPELINE
#-----------------------------



from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# -----------------------------
# 1) Model için dataframe hazırlama sürecim
# -----------------------------
TARGET = "Attrition"  # Hatırlatma: df2_final içinde bu artık 0 ve 1
data = df2_final.copy()

# Eski X, y ayırma kısmını sil ve bunu ekle:
numeric_df = data.select_dtypes(include=[np.number]).copy()

# X'ten hem Attrition'ı hem de o "kopya" olan Attrition_num'ı siliyoruz
# errors='ignore' sayesinde eğer sütun o an orada yoksa hata vermez, temizce geçer.
X = numeric_df.drop(columns=[TARGET, "Attrition_num"], errors='ignore')
y = numeric_df[TARGET]

print(f"Final X shape: {X.shape} | Final y shape: {y.shape}")


# -----------------------------
# 2) Baseline Logistic Regression (Quick Check)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)
print(f"\nBaseline Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# -----------------------------
# 3) Compare Models (Cross-Validation - Accuracy & F1)
# -----------------------------
models = [
    ("LogReg", LogisticRegression(max_iter=1000)),
    ("RF", RandomForestClassifier(random_state=42))
]

print("\nCross-validated Accuracy (5-fold):")
for name, m in models:
    cv_acc = np.mean(cross_val_score(m, X, y, cv=5, scoring="accuracy"))
    print(f"{name}: {cv_acc:.4f}")

# -----------------------------
# 4) GridSearch on RandomForestClassifier
# -----------------------------
rf = RandomForestClassifier(random_state=42)
rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
print("\nGridSearch Best Params:", grid.best_params_)

# Evaluate Final Model
y_pred_rf = best_rf.predict(X_test)
print("\nFinal Model (RF) Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Feature Importance
fi = pd.DataFrame({"feature": X.columns, "importance": best_rf.feature_importances_}).sort_values("importance", ascending=False).head(20)
plt.figure(figsize=(10, 8))
sns.barplot(x="importance", y="feature", data=fi)
plt.title("HR Attrition - Top 20 Feature Importances")
plt.tight_layout()
plt.show(block=True)


# Hata Matrisi Görselleştirme
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Kalıyor", "Gidiyor"])

plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues")
plt.title("Model Tahmin Başarı Tablosu (Confusion Matrix)")
plt.show(block=True)