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

---

## 🚢 Step 0:  İlk ML Model Hazırlığı (Titanic Veri Seti)

Bu tablo, Lojistik Regresyon (Logistic Regression) modeli için veri setindeki metin tabanlı ve gereksiz sütunların nasıl temizlenip sayısal forma dönüştürüldüğünü özetler.

| Adım No. | İşlem Kategorisi | Uygulanan Özellik (Feature) | Yöntem ve Açıklama |
| :---: | :--- | :--- | :--- |
| **1** 🔢 | **Nominal Kodlama** (Text to Numeric) | **Embarked** (Biniş Limanı) 🛳️ | **One-Hot Encoding** (Nominal): Limanlar arasında sıra olmadığı için her liman değeri (`S`, `C`, `Q`) için yeni ikili (binary) sütunlar oluşturuldu. |
| **2** 🏷️ | **Sırasal Kodlama** (Ordinal Encoding) | **AgeGroup** (Yaş Grubu) | **Label Encoding** (Ordinal): Yaş grupları arasında doğal bir sıra olduğu için (Child < Teen < Adult vb.) 0'dan 4'e kadar sayısal değerlere eşlendi (`AgeGroup_num`). |
| **3** ✨ | **Özellik Mühendisliği** (Gruplama) | **Title** (Başlık) | **Nadir Gruplama:** Veri setindeki az sayıda geçen başlıklar (`Mlle`, `Ms`, `Mme`, vb.) "**Other**" kategorisi altında toplandı. Bu, encoding sonrası oluşacak gereksiz sütun sayısını azalttı. |
| **4** 🔢 | **Nominal Kodlama** (Gruplanmış) | **Title** (Başlık) | **One-Hot Encoding** (Nominal): Gruplanmış başlıklar (`Mr`, `Mrs`, `Miss`, `Master`, `Other`) için ikili (binary) sütunlar oluşturuldu. |
| **5** ❌ | **Gereksiz Sütunları Kaldırma** (Removal) | **PassengerId, Name, Ticket** | **Gerekçe:** Tahmin gücü olmayan (ID), ham metin içeren veya rastgele tanımlayıcı olan sütunlar kaldırıldı. |
| **6** 🗑️ | **Yedekleri Kaldırma** (Removal) | **Age, AgeGroup, SibSp, Parch** | **Gerekçe:** Bu sütunlar kullanılarak zaten daha iyi, işlenmiş (engineered) versiyonları oluşturulduğu için kaldırıldılar (yerine **AgeGroup_num** ve **FamilySize** kullanılıyor). |
| **7** ⚖️ | **Korelasyon Nedeniyle Kaldırma** | **Fare** | **Gerekçe:** **Fare**, **Pclass** (Yolcu Sınıfı) ve **Cabin** gibi diğer özelliklerle yüksek oranda ilişkili olduğu için kaldırıldı. |

## 🎉 Final Takeaway: Model Ready!

**Özet Çıkarım:** Modellemeye hazır veri setinde artık sadece **temiz, işlenmiş (engineered) ve tamamen sayısal** özellikler (AgeGroup_num, FamilySize, Title_Mr, Embarked_S vb.) kalmıştır.

Tüm bu hazırlık adımları, modelin **hızlı, tutarlı ve doğru öğrenmesini** sağlamak için kritik öneme sahiptir.

## 🚀 Step 1 & 2: İlk ML Modeli: Veri Bölme Adımları 

<img width="660" height="253" alt="image" src="https://github.com/user-attachments/assets/14ffd280-314c-4e58-bb12-aa111eca7f9e" />


Bu süreç, temizlenmiş veri setini (`df_lr`) alır ve Lojistik Regresyon (Logistic Regression) sınıflandırma modelini kurmak için hazır hale getirir.

| Aşama No. | Aşama Adı (İngilizce Terim) | Yapılan İşlemler | Neden Önemli? (Gerekçe) |
| :---: | :--- | :--- | :--- |
| **1** 🪓 | **Split Features (X) and Target (y)** (Özellikleri ve Hedefi Ayırma) | Veri setindeki tüm girdi sütunları (**Features / X**), tahmin edilmek istenen **çıktı sütunundan (Target / y)** ayrılır. <br>— **X:** Hayatta kalmayı tahmin etmek için kullanılacak sütunlar. <br>— **y:** `Survived` sütunu (tahmin hedefi). | **Veri Sızıntısını (Data Leakage) Önler:** Hedef değişkenin (y) özellik setine (X) sızmasını engeller. Aksi takdirde, model cevabı ezberleyerek gerçek performansı yansıtmayacak şekilde "hile" yapar. |
| **2** ✂️ | **Split into Train/Test** (Eğitim/Test Olarak Bölme) | Ayrılmış X ve y kümeleri, modelin **öğrenmesi** ve **değerlendirilmesi** için iki parçaya bölünür. <br>— **Training Set:** Modeli eğitmek için kullanılır (genellikle %70-%80). <br>— **Test Set:** Modelin genelleme yeteneğini ölçmek için sadece en sonda kullanılır (genellikle %20-%30). | **Genellemeyi Sağlar:** Modelin **görülmemiş** yeni verilere ne kadar iyi tepki verdiğini test eder (gerçek hayatı taklit eder). Modelin eğitimde yapay olarak yüksek skor almasını engeller. |
| **Kontrol** | **Kümelerin Boyutu** | Bölme sonrası, eğitim ve test kümelerindeki sütun sayısı aynı kalır (Örn: (711, 11) ve (178, 11)). Satır sayısı bölünürken, sütunlar korunur. | **Aynı Özellik Uzayı (Same Feature Space):** Modelin doğru çalışması için, hem eğitim hem de test kümelerinin aynı özellik setine sahip olması hayati önem taşır. |


# 📈 Lojistik Regresyon: Katsayılar ve Intercept Özeti

* **Intercept** = Varsayılan Tahmin. Intercept, özellikler dikkate alınmadan önce modelin **başlangıç temel tahminidir**.
* **Katsayılar (Coefficients)** = Özellikler tarafından yapılan ayarlamalar.
    * **Pozitif katsayı** → Özellik, hayatta kalma olasılığını **artırır**.
    * **Negatif katsayı** → Özellik, hayatta kalma olasılığını **azaltır**.

---



## 🧠 Step 3 & 4: ML Modeli Kurulumu ve Eğitimi

Bu tablo, Lojistik Regresyon (Logistic Regression) model nesnesinin, eğitilmiş modele dönüşme sürecini ve elde edilen temel çıktıların (Katsayılar ve Kesişim) yorumunu açıklamaktadır.

| Aşama No. | Aşama Adı (İngilizce Terim) | Yapılan İşlem ve Kod | Açıklama ve Elde Edilen Çıktı |
| :---: | :--- | :--- | :--- |
| **3** 🎯 | **Choose the Algorithm** (Algoritma Seçimi) | **Kullanılan Algoritma:** Lojistik Regresyon (Logistic Regression) - İkili Sınıflandırma için. <br> **Kod:** `LogisticRegression(max_iter=1000)` <br> *(`max_iter` gibi parametreler hiperparametredir.)* | **Boş Nesne:** Bu aşamada model, yalnızca yöntemi bilen **"boş bir algoritma nesnesidir"**. Eğitilmemiştir; hangi özelliklerin önemli olduğunu bilmez. |
| **4** 🏋️ | **Fit the Model** (Modeli Eğitme) | **Kod:** `model.fit(X_train, y_train)` | **Eğitim:** Model, X_train verilerine bakar ve y_train hedefini açıklayan kalıpları bulmak için iç parametrelerini ayarlar. <br> **Sonuç:** Model artık **eğitilmiş bir modeldir** ve yeni yolcular için tahmin yapabilen matematiksel formülü içerir. |
| **Çıktı 1** ➕➖ | **Model Katsayıları** (`model.coef_`) | **Kullanım:** Katsayıların değerlerini kontrol etmek. <br> **Örnek Kod:** `model.coef_` | **Yorum:** Her sayı bir özelliğe karşılık gelir.<br> **Pozitif Katsayı (+):** Özellik, hayatta kalma olasılığını artırır (Örn: **Sex** (kadın=1)).<br> **Negatif Katsayı (-):** Özellik, hayatta kalma olasılığını azaltır (Örn: **Pclass** (büyük sayı = düşük sınıf)).<br> **Büyüklük:** Katsayının mutlak değeri, etkinin gücünü gösterir. |
| **Çıktı 2** 📉 | **Kesişim** (`model.intercept_`) | **Kullanım:** Modelin başlangıç tahminini görmek. <br> **Örnek Kod:** `model.intercept_` | **Yorum:** Kesişim (Intercept), modelin herhangi bir özelliğe bakmadan önceki **başlangıç temel tahminidir**.<br> **Basit Anlamı:** Yolcu hakkında hiçbir şey bilmeseydik elde edeceğimiz **varsayılan hayatta kalma şansı** (log-odds cinsinden) bu değerdir (~2.84). Diğer özellikler bu başlangıç olasılığını yukarı veya aşağı hareket ettirir. |


### Adım 3 → Algoritmayı Seçme (Choose the Algorithm)
* İkili sınıflandırma için basit ama güçlü bir algoritma olan **Lojistik Regresyon**'u seçtik.
* Bu aşamada, `model` yalnızca **boş bir algoritma nesnesidir**; nasıl öğreneceğini bilir, ancak Titanic verilerinden henüz bir şey öğrenmemiştir.

### Adım 4 → Modeli Eğitme (Fit the Model)
* **Fitting = training (eğitme)** demektir. Model, eğitim verilerine (`X_train`, `y_train`) bakar ve her bir özellik için **katsayıları (weights)** öğrenir.
* Eğitimden sonra, model artık "boş" değildir — yeni yolcular için hayatta kalma olasılıklarını tahmin edebilen **matematiksel bir formül** içerir.

## 🔮Step 5: ML Modeli Tahmin Aşaması (Make Predictions)

Model eğitimini tamamladıktan sonra, test verisi üzerinde tahminler yapmak ve bu çıktıları analiz etmek için aşağıdaki adımlar uygulanır.

| Aşama No. | Yapılan İşlem (Metot) | Çıktı Tipi ve Kod Kullanımı | Temel Formül ve Yorumlama |
| :---: | :--- | :--- | :--- |
| **1** 🏷️ | **Sınıf Etiketlerini Tahmin Etme** | **Çıktı:** İkili sınıf etiketi (0 veya 1). <br> **Kod:** `.predict(X_test)` | **Yorum:** Tahminin eşik değerine (varsayılan 0.5) göre nihai evet/hayır kararıdır. <br>— **0:** Hayatta kalmadı. <br>— **1:** Hayatta kaldı. |
| **2** 📈 | **Olasılıkları Tahmin Etme** | **Çıktı:** Her bir sınıf için olasılıklar. İki sütunlu dizi verir. <br>— **Sütun 0:** Sınıf 0'a (Hayatta kalmama) ait olma olasılığı. <br>— **Sütun 1:** Sınıf 1'e (Hayatta kalma) ait olma olasılığı. <br> **Kod:** `.predict_proba(X_test)` | **Yorum:** Lojistik Regresyon'un temeli **olasılıklara** dayanır. Olasılıklar, modelin tahminindeki **güven düzeyini** gösterir. <br> *Örn: [0.10, 0.89] → %89 hayatta kalma şansı.* |
| **3** 📖 | **Sonuçları Birleştirme ve Okuma** | **Çıktı:** Üç temel bilgi sütunu: Actual, Predicted, Prob_Survived. | **Okuma:** <br>— **Actual:** Gerçekleşen sonuç. <br>— **Predicted:** Modelin kararı (eşik sonrası). <br>— **Prob_Survived:** Modelin karar vermek için kullandığı temel olasılık. Bu, modelin **emin olduğu** veya **sınırda kaldığı** durumları (0.5 civarı) görmeyi sağlar. |
| **4** ⚖️ | **Eşiğin Rolü (Threshold)** | **Varsayılan Eşik:** 0.5. **Esas:** False Positive (Yanlış Pozitif) veya False Negative (Yanlış Negatif) maliyetine göre eşiğin manuel ayarlanması. <br> **Kod:** `.predict_proba()` çıktısı kullanılarak manuel kontrol edilir. | **Kural:** <br>— Sınıf $\ge 0.5 \rightarrow$ Tahmin 1. <br>— Olasılık $< 0.5 \rightarrow$ Tahmin 0. <br><br> **Ayarlama:** <br>— **Düşük Eşik:** **Recall'ı** (Hassasiyeti) artırır (daha fazla vaka yakalar). <br>— **Yüksek Eşik:** **Precision'ı** (Kesinliği) artırır (daha muhafazakâr tahmin yapar). |


## ⚖️ Step 6: ML Modeli Değerlendirme Aşaması (Evaluate)

Bu tablo, ilk sınıflandırma modelimizin (Lojistik Regresyon) performansını ölçmek için kullanılan temel metrik olan **Doğruluğu (Accuracy)** ve bu metriğin yorumunu özetler.

| Kategori | Açıklama ve Metrik | Tanım ve Yorumlama | Kritik Notlar ve Çıkarımlar |
| :--- | :--- | :--- | :--- |
| **Temel Amaç** | **Değerlendirme** (Evaluate) | Modelin test seti üzerindeki tahminleri ne kadar iyi gerçekleştirdiğini kontrol etme. | Model, görülmemiş verilere karşı ne kadar iyi **genelliyor** (generalizes)? |
| **Temel Metrik** | **Doğruluk** (Accuracy) | **Soru:** "Test setindeki tüm yolcular arasında, model doğru tahmin oranını ne kadar tutturdu?" | **Formül:** <br> $$\text{Doğruluk} = \frac{\text{Doğru Tahmin}}{\text{Toplam Tahmin}}$$ |
| **Yorumlama Örneği** | Doğruluk = **0.81** | Model, test setindeki yolcuların **%81'inin** hayatta kalıp kalmadığını doğru tahmin etti. (Kabaca, her 10 tahminden 8'i doğruydu.) | Bu model, **%81 başarı oranıyla** çalışan ilk sınıflandırma modelimizdir. |
| **Gelecek Kritik** | **Doğruluğun Sınırları** | Doğruluk iyi bir başlangıç olsa da, hikayenin tamamı değildir. Sınıf dengesizliği (örneğin, herkesin öldüğü bir senaryo) olduğunda, sadece "ölüm" tahmin eden basit bir model bile yüksek doğruluk alabilir, ancak **işlevsiz** olur. | Daha derin bir analiz için daha sonra **precision, recall, F1** ve **confusion matrices** (karmaşıklık matrisleri) kullanılacaktır. |
| **Ana Çıkarım** 🎉 | **Key Takeaways** | Lojistik Regresyon ilk sınıflandırma modelimizi kurmuştur. | **6 adımlık iş akışı** (veri → özellikler → model → tahminler → değerlendirme) başarıyla tamamlanmıştır. Lojistik Regresyon basittir ancak güçlü bir **temel performans** (**strong baseline**) sağlar. |
