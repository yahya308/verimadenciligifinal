# BLM308 Veri Madenciliği - Final Projesi

**Hazırlayan:** Yahya Baltacı
**Öğrenci No:** 231041016
**Dönem:** Bahar 2026

## Proje Hakkında
Bu proje, veri madenciliği süreçlerini (CRISP-DM) uçtan uca uygulayarak, klinik verilere dayalı bir hastalık tahmin modeli geliştirmeyi amaçlamaktadır. Proje kapsamında "Pima Indians Diabetes" veri seti kullanılmış olup, hastaların diyabet riski taşıyıp taşımadığı üç farklı makine öğrenmesi algoritması (J48 Karar Ağacı, Naive Bayes ve IBk k-En Yakın Komşu) ile test edilmiş ve performansları karşılaştırılmıştır. 

WEKA sonuçlarının tutarlılığını doğrulamak amacıyla Naive Bayes modeli Python (scikit-learn) ile de test edilmiştir.

## Dosya ve Klasör Yapısı
Proje dizini aşağıdaki dosyalardan oluşmaktadır:

* `verimadenciligi.pdf` : Projenin tüm araştırma, ön işleme, modelleme ve iş değeri aşamalarını içeren detaylı LaTeX çıktısı (Final Raporu).
* `main.tex` : Raporun oluşturulduğu orijinal LaTeX kaynak kodu.
* `python_karsilastirma.py` : WEKA'daki Naive Bayes sonuçlarını doğrulamak için yazılmış Python (scikit-learn) betiği.
* `sunumvideo.mp4` : Projenin WEKA üzerinde nasıl çalıştırıldığını ve sonuçların nasıl elde edildiğini gösteren yerel demo videosu.
* `veri/diabetes.arff` : Projede kullanılan Pima Indians Diabetes veri setinin WEKA formatındaki hali.
* `kod/j48.txt` : WEKA üzerinden alınan J48 Karar Ağacı modeline ait deney sonuçları ve karmaşıklık matrisi.
* `kod/NaiveBayes.txt` : WEKA üzerinden alınan Naive Bayes modeline ait deney sonuçları ve karmaşıklık matrisi.
* `kod/IBK.txt` : WEKA üzerinden alınan IBk (k=1) modeline ait deney sonuçları ve karmaşıklık matrisi.

## Bağlantılar
* **GitHub Deposu:** https://github.com/yahya308/verimadenciligifinal
* **YouTube Demo Videosu:** https://youtu.be/o6OsWhO8YJI

## Modellerin Çalıştırılması
**1. WEKA Üzerinden Çalıştırma:**
* WEKA Explorer arayüzünü açın.
* `Preprocess` sekmesinden `veri/diabetes.arff` dosyasını yükleyin.
* `Classify` sekmesine geçin ve test seçeneğini `10-fold cross-validation` olarak ayarlayın.
* Sırasıyla `trees -> J48`, `bayes -> NaiveBayes` ve `lazy -> IBk` algoritmalarını seçerek modelleri çalıştırın.

**2. Python Üzerinden Çalıştırma:**
* Terminal veya komut satırında proje ana dizinine gidin.
* `pip install pandas scikit-learn scipy` komutuyla gereksinimleri yükleyin (eğer yüklü değilse).
* `python python_karsilastirma.py` komutuyla betiği çalıştırın.