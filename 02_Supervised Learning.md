# 🧠 Supervised Learning

<img width="654" height="368" alt="image" src="https://github.com/user-attachments/assets/40ebd4b1-bafc-4198-9c59-7745c0fda5d7" />


* **Supervised Learning** 🧑‍🏫 = Etiketli verilerden öğrenme (Girdi → Çıktı).

* **İki temel tip:**
    * **Classification** 🎯 → Kategorik çıktı (Örn: Evet/Hayır, Kırmızı/Mavi).
    * **Regression** 📈 → Sayısal çıktı (Örn: Fiyat, Sıcaklık).

* **Temel İş Akışı (Workflow):** Veriyi Ayır (split) → Eğit (fit) → Tahmin Et (predict) → Değerlendir (evaluate).

* `sklearn` (scikit-learn) 🐍 tüm algoritmalar arasında bu süreci **tutarlı** hale getirir.

## 🧠 Summary: Supervised Learning Process

Gözetimli Öğrenme (Supervised Learning) projelerinin temel adımları aşağıdadır:

1.  **Veriyi X ve y olarak ayırın** (Split data into X and y) 🪓:
    * **X = features** (Tahmin yapmak için kullandığınız girdiler).
    * **y = target** (Tahmin etmek istediğiniz etiket/cevap).

2.  **Eğitim/Test olarak bölme** (Train/test split) ✂️:
    * **Train (Eğitim):** Modelin öğrendiği veri kümesi.
    * **Test (Test):** Modelin performansının ölçüldüğü küme (görülmemiş veri).

3.  **Bir algoritma seçin** (Choose an algorithm):
    * Örn: Logistic Regression, Decision Tree, vb.

4.  **Modeli Eğitin** (Fit the model) 🧠:
    * `.fit(X_train, y_train)` (Eğitim kümesinden öğrenir).

5.  **Tahmin Yapın** (Predict) 🔮:
    * `.predict(X_test)` (Görülmemiş test verisi üzerinde tahminler yapar).

6.  **Değerlendirin** (Evaluate) ⚖️:
    * Doğru metriklerle performansı kontrol edin:
        * **Classification (Sınıflandırma):** accuracy, precision, recall, F1, AUC.
        * **Regression (Regresyon):** MAE, RMSE, R².

---

> ✨ **Golden Rule (Altın Kural):** Yalnızca **eğitim (train)** verisi üzerinde eğitiminizi yapın (fit), ardından **test** verisi üzerinde dönüştürme (transform), tahmin (predict) ve değerlendirme (evaluate) işlemlerini yapın.
---
---

# 🧑‍🏫 What is Supervised Learning?

In **Supervised Learning**, we train a model using data where the correct answer (label) is already known. The model learns a functional relationship by mapping inputs to known outputs.

* **Inputs (Features) 📊:** The data we use to make predictions.
* **Output (Target/Label) 🎯:** The known value we want the model to predict.
* **The Model's Goal:** Learn the mapping between inputs and outputs.

---

## Types of Supervised Learning Tasks

Supervised learning problems are divided into two main categories based on the nature of the predicted output:

### 1. Classification (Predict Categories)
The model predicts a **discrete, categorical output** (a class or label).
* **Examples:**
    * Spam vs Not Spam.
    * Survived vs Did Not Survive (Titanic dataset).

### 2. Regression (Predict Numbers)
The model predicts a **continuous, numerical output** (a real value).
* **Examples:**
    * Predict house prices.
    * Predict passenger age if missing.

---

## The Supervised Learning Workflow 🧠

Now that you know what supervised learning is, let's see how it actually works in practice.

Think of it like training a student for an exam:
1.  You give them examples with answers (**training data**).
2.  They learn the patterns (**fit a model**).
3.  You test them with new questions they haven’t seen before (**test data**).
4.  You check how well they did (**evaluation**).

Machine learning follows the exact same process. We'll break it into 6 clear steps — from splitting the data to evaluating the model — so you can apply it to any dataset.

**Remember, all these steps are done after preprocessing your data!** 🛠️


## 🧠 Supervised Learning İş Akışı: Adımlar ve Gerekçeler

Bu tablo, Gözetimli Öğrenme projesindeki temel adımları, her adımın amacını ve uygulanan yöntemleri özetler.

| Aşama (İngilizce) | Amaç ve Açıklama | Teknik Detaylar (The "How") | Neden Önemli? (Why it Matters) |
| :--- | :--- | :--- | :--- |
| **Split Data** 🪓 | Veri setini **Girdiler (Features/X)** ve **Çıktı (Target/y)** olmak üzere ikiye ayırma. | **Features (X):** Tahmin için kullanılan girdi değişkenleri (Örn: Age, Fare, Pclass, encoded Sex). **Target (y):** Tahmin etmek istenen çıktı (Örn: Survived veya fiyat). | Girdilerinizi cevap anahtarından ayrı tutar. **Veri sızıntısını (Data Leakage)** önler (y'yi veya olay sonrası bilgileri X içine asla dahil etmeyin). |
| **Split into Train/Test** ✂️ | Modeli **train (eğitim)** verisinde öğrenmesi ve **test** verisinde (görülmemiş) değerlendirilmesi için bölme. | Veri setini genellikle %70-80 eğitim ve %20-30 test olarak böleriz. | **Gerçek hayatı taklit eder** (yeni, görülmemiş veri). İyimser ve yanıltıcı sonuçları engeller. |
| **Choose an Algorithm** 🎯 | Problem tipinize ve veri yapınıza en uygun modeli seçme. | **Sınıflandırma (Classification):** Logistic Regression, Decision Tree, Random Forest, SVM. **Regresyon (Regression):** Linear Regression, Decision Tree/Random Forest. **Tavsiye:** Daha karmaşık modellere geçmeden önce basit bir başlangıç modeli (**Baseline**) ile başlayın. | Farklı modeller farklı varsayımlar (doğrusallık, etkileşimler) yapar ve farklı ödünleşimlere (hız, yorumlanabilirlik) sahiptir. |
| **Fit the Model** 🧠 | Modelin eğitim verisinden öğrenmesi. (Preprocessing'deki transformer'lar ile aynı mantık). | `.fit(X_train, y_train)` metodu, modelin parametrelerini (ağırlıklar, bölme noktaları vb.) öğrenmesini sağlar. | Yalnızca eğitim kümesi, modelin (ve herhangi bir ön işlemenin) bilgi edinmesini sağlamalıdır (sızıntıyı önlemek için). |
| **Predict** 🔮 | Modelin görülmemiş test verisi üzerinde kullanılması. | Sayı/sınıf tahminleri için `.predict(X_test)` kullanılır. Sınıflandırma olasılıkları için `.predict_proba(X_test)` kullanılır (eşik ayarı için faydalıdır). | Modelin yeni veriyi nasıl genellediğini gösterir. |
| **Evaluate** ⚖️ | Tahminleri, görev türünüze uygun metriklerle `y_test` ile karşılaştırma. | **Sınıflandırma Metrikleri:** Accuracy (genel doğruluk), Precision/Recall & F1 (sınıf dengesizliğine duyarlı), ROC AUC / PR AUC (sıralama kalitesi), Confusion Matrix (hata tipleri). **Regresyon Metrikleri:** MAE (ortalama mutlak hata), RMSE (büyük hataları cezalandırır), R² (açıklanan varyans). | Metrikler, modelinizin iş hedefinize ne kadar iyi ulaştığını objektif olarak ölçmenizi sağlar. |

