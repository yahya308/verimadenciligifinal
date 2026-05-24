import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore') # Ekrandaki gereksiz uyarıları gizle

# 1. Veri setini yükle (Proje ana klasöründen çalıştırıldığı varsayımıyla)
file_path = 'veri/diabetes.arff'

try:
    print(f"'{file_path}' dosyası okunuyor...")
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
except FileNotFoundError:
    print(f"\nHATA: '{file_path}' bulunamadı!")
    print("Lütfen terminalde (komut satırında) projenin ana klasöründe olduğunuza emin olun.")
    exit()

# Scipy ARFF okuyucusu stringleri byte olarak (b'tested_positive') okur, bunu düzeltelim:
df['class'] = df['class'].str.decode('utf-8')

# Bağımsız değişkenler (X) ve hedef değişken (y)
X = df.drop('class', axis=1)
y = df['class']

# Hedef değişkeni scikit-learn'ün anlayacağı 1 ve 0 formatına çevir
y_binary = y.map({'tested_positive': 1, 'tested_negative': 0})

# 2. Model ve 10-Katlı Çapraz Doğrulama (10-Fold Cross-Validation)
model = GaussianNB()
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Ölçülecek metrikler
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

print("-" * 50)
print("Python (scikit-learn) Naive Bayes Test Sonuçları")
print("-" * 50)

# 3. Modeli Eğit ve Test Et
scores = cross_validate(model, X, y_binary, cv=cv, scoring=scoring)

# 4. Sonuçları Ekrana Yazdır
acc = scores['test_accuracy'].mean() * 100
prec = scores['test_precision'].mean()
rec = scores['test_recall'].mean()
f1 = scores['test_f1'].mean()
roc = scores['test_roc_auc'].mean()

print(f"Accuracy (Doğruluk):  %{acc:.2f}")
print(f"Precision (Kesinlik): {prec:.3f}")
print(f"Recall (Duyarlılık):  {rec:.3f}")
print(f"F1 Score:             {f1:.3f}")
print(f"ROC-AUC:              {roc:.3f}")

print("\n--- WEKA vs Python Karşılaştırması ---")
print(f"WEKA Accuracy:   %76.30  |  Python Accuracy:   %{acc:.2f}")
print(f"WEKA ROC-AUC:    0.819   |  Python ROC-AUC:    {roc:.3f}")
print("\nNot: Python ve WEKA'nın veriyi 10 parçaya bölerken kullandığı rastgelelik")
print("(random seed / shuffling) algoritmaları farklı olduğu için küsuratlarda ufak sapmalar olması doğaldır.")