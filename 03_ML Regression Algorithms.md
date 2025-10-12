
# 📉 ML Regresyon Algoritmaları: Regresyona Giriş

---

## 🎯 Regresyon Nedir?

<img width="681" height="353" alt="image" src="https://github.com/user-attachments/assets/56d5922a-0b52-45f1-80cd-fbb999ac63c6" />
 <img width="618" height="355" alt="image" src="https://github.com/user-attachments/assets/78c3bbfd-ee6a-4e12-b98b-c609b11fc5c4" />

| Kavram | Açıklama |
| :--- | :--- |
| **Öğrenme Tipi** | **Denetimli Öğrenme** (Supervised Learning) |
| **Amaç Değişkeni (Target)** | **Sürekli** (Continuous) veya **Sayısal** bir değerdir (kategorik olmayan). |
| **Amacı** | Girdi özelliklerine dayanarak **sayısal bir değerin tahmin edilmesi**. |
| **Örnek Soru** | "Bu evin satış fiyatı ne olacak?" veya "Gelecek ay kaç birim satış yapacağız?" |
| **Özet** | **Sınıflandırma** kategorileri tahmin eder, **Regresyon** ise **sayıları** tahmin eder. |

# 🚀 Sınıflandırma (Classification) vs Regresyon (Regression) Karşılaştırması

| 🏷️ Görev/Özellik | 🎯 Sınıflandırma (Classification) | 📉 Regresyon (Regression) |
| :--- | :--- | :--- |
| **Hedef Değişken Tipi** | **Kategorik** (Categorical) | **Sürekli** (Continuous) |
| **Örnek Değerler** | Evet/Hayır (Yes/No), Sahte/Gerçek (Fraud/Not Fraud), Müşteri/Müşteri Değil | Fiyat (Price), Satış Miktarı (Sales), Yaş (Age), Sıcaklık |
| **Örnek Soru** | "Yarın yağmur yağacak **mı**?" (Evet veya Hayır cevabı) | "Yarın ne kadar yağmur **yağacak**?" (Sayısal bir değer) |
| **Örnek Algoritmalar** | **Lojistik Regresyon** (Logistic Regression) | **Lineer Regresyon** (Linear Regression) |
| | Karar Ağaçları (Decision Trees) | Karar Ağaçları (Decision Trees) |
| | Rastgele Orman (Random Forest) | Rastgele Orman (Random Forest) |
| | k-En Yakın Komşu (kNN) | k-En Yakın Komşu (kNN) |
| | Destek Vektör Makineleri (SVM) | |


---

## 📌 Lojistik Regresyon (Sınıflandırma) ve Lineer Regresyon (Regresyon) Karşılaştırması

| Özellik | Lojistik Regresyon (Sınıflandırma) | Lineer Regresyon (Regresyon) |
| :--- | :--- | :--- |
| **Hedef** | **Kategorileri** tahmin eder (örn: spam/spam değil). | **Sürekli** bir değeri tahmin eder (örn: evin fiyatı). |
| **Çıktı** | 0 ile 1 arasında bir **olasılık**, daha sonra bir sınıfa dönüştürülür. | Herhangi bir değer olabilen **bir sayı** (belirli bir aralıkla sınırlı değildir). |
| **Örnek** | Bir müşteri bir ürün satın alacak mı? (Evet veya Hayır) | Bu evin fiyatı ne olacak? |
| **Benzerlik** | Her ikisi de girdi özellikleri ile hedef değişken arasındaki en iyi ilişkiyi bulmaya çalışır. | |

---

## 🏡 Regresyonun Gerçek Dünya Kullanım Senaryoları

| 🖼️ Senaryo | Açıklama |
| :--- | :--- |
| **🏠 Ev Fiyatları** | Boyut, konum ve durum gibi özelliklerden mülk fiyatlarının tahmin edilmesi. |
| **💊 Tıbbi Maliyetler** | Demografik ve sağlık verilerine dayanarak hastaların sağlık harcamalarının tahmin edilmesi. |
| **📊 Satış Tahmini** | Geçmiş verilere dayanarak gelecekteki satış hacminin veya gelirin tahmin edilmesi. |
| **🌡️ Hava Tahmini** | Sıcaklık veya yağış miktarının tahmin edilmesi. |
| **Kullanım Alanı** | Sayısal bir değerin tahmin edilmek istendiği **her yer**. |

---

## ⚙️ Regresyon İş Akışı (Supervised Learning)

| Adım | Açıklama |
| :--- | :--- |
| **1. Veri Ayırma** | Verileri **özellikler (X)** ve **hedef (y)** olarak ayırın. |
| **2. Eğitim/Test Bölme** | Görülmeyen veriler üzerinde test yapabilmek için veriyi eğitim ve test setlerine ayırın. |
| **3. Algoritma Seçimi** | Bir algoritma seçin (örn: **Lineer Regresyon**). |
| **4. Modeli Eğitme (Fit)** | Eğitim verilerinden kalıpları öğrenmek için modeli eğitin. |
| **5. Tahmin Etme (Predict)** | Yeni/test verileri üzerinde tahminler yapın. |
| **6. Değerlendirme (Evaluate)** | **Regresyon metriklerini** kullanın (MSE, RMSE, MAE, R²). |
| **❗ Temel Fark** | Regresyonda, **hedef (y) bir sayıdır**, bir kategori değil. |

---

## 📏 İlk Regresyon Modeliniz: Lineer Regresyon

| Kavram | Açıklama |
| :--- | :--- |
| **Amacı** | Verileriniz arasına **düz bir çizgi** (veya daha yüksek boyutlarda bir düzlem/hiper-düzlem) oturtmaya çalışır. |
| **Ne Yapar?** | Bu çizgi, girdi özellikleri (örn: ev boyutu, X ekseni) ile tahmin etmeye çalıştığınız sürekli değer (örn: ev fiyatı, Y ekseni) arasındaki **ilişkiyi** yakalamaya çalışır. |
| **Çalışma Prensibi** | Model, tahmin edilen ve gerçek değerler arasındaki **mesafeyi en aza indiren** (yani tahminlerin hatasını en aza indiren) çizgiyi (veya düzlemi/hiper-düzlemi) bulur. |

----

# 📉 Regresyona Giriş II: Regresyon Derinlemesine

<img width="611" height="320" alt="image" src="https://github.com/user-attachments/assets/20a31df5-136b-495c-a22e-6f03ae416d41" />
---

## 🔎 Lineer Regresyon Nasıl Çalışır?

Lineer regresyonda amaç, verilere en iyi uyan **düz çizgiyi** bulmaktır.

### 📝 Tek Özellikli Lineer Denklemi (2 Boyutlu)

| Sembol | Açıklama |
| :--- | :--- |
| **$y = mx + b$** | **Düz çizginin temel denklemidir.** |
| **$y$** | Tahmin ettiğimiz değer (**Hedef Değişken**). Örn: Ev fiyatı. |
| **$x$** | Girdi özelliği (**Bağımsız Değişken**). Örn: Evdeki oda sayısı. |
| **$m$** | **Eğim (Slope)**. $x$'teki bir birimlik artışın $y$'de ne kadar değişikliğe yol açtığını gösterir. Örn: Her ek oda ile ev fiyatının ne kadar arttığı. |
| **$b$** | **Kesişim (Intercept)**. $x=0$ olduğunda $y$'nin başlangıç değeri. Örn: Ev boyutu sıfır olduğunda varsayılan başlangıç fiyatı. |
| **Amaç** | **$y$'yi $x$'ten en iyi şekilde tahmin etmek için $m$ ve $b$ değerlerini bulmaktır.** |

### 📊 Çok Özellikli Lineer Denklem

Tek bir özellik yerine birden çok özellik olduğunda (ki bu daha yaygındır), formül bir **hiper-düzlem** haline gelir:

$$y = b_0 + b_1x_1 + b_2x_2 + \dots + b_nx_n$$

---

## ⚠️ Lineer Regresyonun Varsayımları ve Sınırlamaları

Lineer regresyonun iyi çalışması ve güvenilir sonuçlar vermesi için bazı temel varsayımların geçerli olması gerekir.

### 📋 Temel Varsayımlar

| Emoji/Görsel | Varsayım | Açıklama | İhlal Durumunda |
| :--- | :--- | :--- | :--- |
| **📏** | **Lineerlik (Linearity)** | Özellikler ($x$) ile hedef değişken ($y$) arasındaki ilişkinin **kabaca doğrusal** olması. | Polinom regresyon veya ağaç tabanlı modeller düşünülmelidir. |
| **🔄** | **Bağımsızlık (Independence)** | Gözlemlerin **birbirinden bağımsız** olması (Hata terimleri arasında ilişki olmaması). | Aynı kişiden tekrarlanan ölçümler bu durumu bozabilir. |
| **🔔** | **Hataların Normalliği (Normality of Errors/Residuals)** | Tahmin hatalarının (gerçek değer - tahmin edilen değer) **normal dağılıma** sahip olması. | Güven aralıklarını ve istatistiksel testleri etkileyebilir. |
| **⚖️** | **Homoscedasticity (Hataların Eşit Varyansı)** | Hataların varyansının, özellik değerleri boyunca **sabit** olması. | Hatalar, özellik değerleri yükseldikçe artıyorsa tahminler güvenilmez olabilir. |
| **↔️** | **Çoklu Doğrusallık Olmaması (No Multicollinearity)** | Özelliklerin **birbirleriyle yüksek oranda ilişkili** olmaması. | Yüksek korelasyon, katsayıları yorumlamayı zorlaştırır ve kararsız hale getirir. |
| ** outliers** | **Aykırı Değer Olmaması (No Outliers)** | Veri setinde regresyon çizgisini ağır şekilde çarpıtabilecek **aşırı değerlerin** olmaması. | Aykırı değerler, modelin verilerin çoğunluğuna kötü uyum sağlamasına neden olabilir. |

### ❌ Sınırlamalar ve Çözümler

* **Varsayım İhlali:** Varsayımlar ihlal edilirse, lineer regresyon **eksik uyum** sağlayabilir (kalıpları kaçırabilir) veya **yanıltıcı katsayılar** verebilir.
* **Çözüm:** Pratikte, bu varsayımlar karşılanmadığında **özellik mühendisliği**, **dönüşümler** (transformations) veya **daha esnek/düzenlileştirilmiş** (regularized) modeller (örn: karar ağaçları) kullanmak gerekebilir.

---

## 🔑 Önemli Çıkarım

Regresyon, sürekli çıktıları tahmin etmemize yardımcı olur. **Lineer Regresyon** en basit şeklidir ancak belirli varsayımlara dayanır. Bu varsayımlar geçerli değilse, daha sonraki derslerde **daha sağlam (robust)** modelleri keşfedeceğiz.

# 📈 Regresyon Modellerini Değerlendirme

Regresyon modelimiz sürekli değerleri (örneğin gelir veya ev fiyatları) ne kadar iyi tahmin ediyor? Bu soruyu yanıtlamak için, modelin tahminlerinin gerçek değerlerden ne kadar uzakta olduğunu ölçen özel metrikler kullanırız.

---

## 🔑 Temel Regresyon Metrikleri

| 🖼️ Metrik | Tanım ve Amaç | Formül/Hesaplama | Artıları (✅) ve Eksileri (❌) |
| :--- | :--- | :--- | :--- |
| **1. Hata Kareleri Ortalaması (Mean Squared Error - MSE)** | **Tanım:** Tahmin edilen ve gerçek değerler arasındaki **farkların karesinin ortalamasıdır.** | $$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$ | ✅ **Büyük hataları ağır şekilde cezalandırır.** (Büyük hataların maliyetli olduğu durumlarda iyidir.)<br> ✅ Modelleri karşılaştırmak için iyi bir metriktir.<br> ❌ **Yorumlaması zordur** (birimi, hedef değişkenin biriminin karesidir, örn: "kare dolar"). |
| **2. Ortalama Mutlak Hata (Mean Absolute Error - MAE)** | **Tanım:** Tahmin edilen ve gerçek değerler arasındaki **mutlak farkların ortalamasıdır.** | $$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat{y}_i|$$ | ✅ **Yorumlaması çok kolaydır.** (Birim, hedef değişkenle aynıdır, örn: "Ortalama $4,000 yanılma").<br> ❌ **Büyük hataları MSE kadar vurgulamaz** (aykırı değerlere karşı daha az hassastır). |
| **3. Hata Kareleri Ortalamasının Karekökü (Root Mean Squared Error - RMSE)** | **Tanım:** **MSE'nin kareköküdür.** MSE'yi hedef değişkenle aynı birime geri dönüştürür. | $$\text{RMSE} = \sqrt{\text{MSE}}$$ | ✅ **Yorumlanabilir** (Birim hedef değişkenle aynıdır).<br> ✅ Hala **büyük hataları cezalandırır** (MAE'den daha fazla).<br> ✅ Çoğu zaman tercih edilen metrik ("tipik tahmin hatası"). |
| **4. R-Kare ($R^2$) / Belirleme Katsayısı** | **Tanım:** Modelin, hedef değişkendeki varyasyonun **ne kadarını açıkladığını** ölçer. | $$R^2 = 1 - \frac{\text{Modelin Kare Hata Toplamı}}{\text{Toplam Varyans}}$$ | ✅ Modelin, **ortalama tahmini kullanmaktan ne kadar daha iyi** olduğunu gösterir.<br> ✅ 1'e yakın olması mükemmel uyum demektir.<br> ❌ Negatif olabilir (model, ortalama tahminden daha kötüyse). |

---

## 💡 Metrikler Nasıl Yorumlanır?

| Metrik | İdeal Değer | Anlamı | Odaklanılması Gereken Durum |
| :--- | :--- | :--- | :--- |
| **MAE, MSE, RMSE** | **0'a Yakın** (Daha Düşük Daha İyi) | Tahminler gerçek değerlere çok yakındır. | Hatanın mutlak büyüklüğü ve/veya büyük hataların maliyeti önemlidir. |
| **$R^2$** | **1'e Yakın** (Daha Yüksek Daha İyi) | Model, hedef değişkendeki varyasyonun büyük bir bölümünü açıklar. | Modelin, genel veriye ne kadar iyi uyum sağladığını görmek istendiğinde. |

### 🧭 Hangi Metriğe Odaklanmalı?

* **MAE:** Modelin ortalama hatasını kolayca açıklamak istiyorsanız ("Tahminimiz ortalama $X$ kadar yanılıyor."). Aykırı değerler model performansını çarpıtıyorsa daha iyi bir seçimdir.
* **MSE / RMSE:** Büyük tahmin hatalarının (aşırı maliyetli olduğu veya kritik olduğu durumlar) daha fazla ağırlık taşımasını istiyorsanız. RMSE, yorumlanabilir olması nedeniyle MSE'ye göre daha sık kullanılır.
* **$R^2$:** Modelinizin, basit bir ortalama tahminine kıyasla ne kadar bilgilendirici olduğunu (varyansı ne kadar açıkladığını) anlamak için.

**Ana Çıkarım:** Regresyonda **doğruluk (accuracy)** anlamsızdır. Modelinizin performansının tam bir resmini vermek için genellikle **birden fazla metrik** raporlamanız gerekir.



# 📈 Regresyon Modelleri: Polinom Regresyon

Sınıflandırmada olduğu gibi, regresyon görevleri için de veri yapısına bağlı olarak güçlü ve zayıf yönleri olan birçok farklı algoritma bulunmaktadır.

---

## 1️⃣ Lineer Regresyon (Tekrar)

| 🖼️ Özellik | Açıklama |
| :--- | :--- |
| **Model Tipi** | En basit regresyon modeli. |
| **Amacı** | Veriler arasına **düz bir çizgi** oturtmak. |
| **İdeal Kullanım** | Özellikler ve hedef arasındaki ilişkinin **yaklaşık olarak doğrusal** olduğu durumlar. |
| **✅ Artıları** | Basit ve **yorumlanabilir** (yorumlaması kolay). Hızlı eğitim ve tahmin süreleri. |
| **❌ Eksileri** | Doğrusal olmayan (non-linear) desenlerle başa çıkmakta zorlanır. Aykırı değerlere ve çoklu doğrusallığa (multicollinearity) karşı hassastır. |

---

## 2️⃣ 〰️ Polinom Regresyon (Polynomial Regression)

 <img width="480" height="369" alt="image" src="https://github.com/user-attachments/assets/e8e55e17-ba83-4082-91e0-6f511d40f1c2" />

| 🖼️ Özellik | Açıklama |
| :--- | :--- |
| **Model Tipi** | **Lineer Regresyonun bir uzantısıdır**; ancak eğri ilişkileri yakalar. |
| **Amacı** | Verilerdeki **eğrileri (non-linear) yakalamak** için $x^2, x^3, \dots$ gibi **polinom terimleri** ekler. |
| **Örnek** | Dondurma satışı ile sıcaklık arasındaki eğrisel ilişkiyi modellemek. |
| **✅ Artıları** | **Doğrusal olmayan (non-linear) ilişkileri** yakalayabilir. Uygulaması nispeten hala basittir (temelde dönüştürülmüş veriler üzerinde Lineer Regresyon çalıştırılır). |
| **❌ Eksileri** | Polinom derecesi çok yüksek olursa **Aşırı Uyum (Overfitting)** riski taşır. Lineer Regresyondan daha az yorumlanabilir. |

### 🎓 Polinom Derecesinin Etkisi

| Derece | Görsel Temsil | Açıklama | Risk |
| :--- | :--- | :--- | :--- |
| **1. Derece** | Düz çizgi (Lineer Regresyon) | En basit model. | Düz bir ilişki yoksa **eksik uyum** (underfitting) yapabilir. |
| **2. Derece** | Hafif eğri (Parabol) | Eğriselliği yakalamaya başlar. Verilere daha iyi uyum sağlayabilir. | |
| **Yüksek Derece (örn. 4)** | "Zikzaklı" veya "Kıvrımlı" çizgi | Çok karmaşık desenleri yakalar ve veri noktalarına çok yakın geçer. | **Aşırı Uyum (Overfitting)**: Verideki gürültüyü (noise) bile öğrenir, yeni verilerde performansı düşer. |

> **💡 Önemli Çıkarım:** Daha yüksek dereceli polinomlar, daha karmaşık desenleri yakalayabilir, ancak derece çok yüksek olursa model verideki **gürültüyü** öğrenerek aşırı uyum yapma riski taşır.

### 🛠️ Polinom Regresyon İş Akışı

1. **Özellik Ölçeklendirme (Scaling):** Özellikler ölçeklendirilir. (Önemlidir, çünkü polinom terimleri sayıları çok büyütebilir.)
2. **Polinom Özellik Oluşturma:** Belirlenen dereceye kadar polinom terimleri ($x^2$, $x^3$, etkileşim terimleri vb.) oluşturulur.
3. **Lineer Regresyon Eğitimi:** **Bu yeni özellikler** üzerinde standart bir Lineer Regresyon modeli eğitilir.
4. **Tahmin ve Değerlendirme:** Tahminler yapılır ve sonuçlar değerlendirilir.

# 🌳 Regresyon Modelleri: Ağaç Tabanlı Yöntemler

Lineer ve Polinom Regresyonun aksine, ağaç tabanlı modeller veriyi doğrusal olmayan şekilde bölgelere ayırarak tahminler yapar.

---

## 3️⃣ Karar Ağacı Regresyonu (Decision Tree Regression)

<img width="702" height="255" alt="image" src="https://github.com/user-attachments/assets/3724cf86-2549-4ba1-baad-c74d7c2d312a" />


| 🖼️ Özellik | Açıklama |
| :--- | :--- |
| **Model Tipi** | Veriyi ikili kararlarla (split) sürekli olarak küçük gruplara böler. |
| **Amacı** | Her bir son (yaprak/leaf) grubun **ortalama hedef değerini** tahmin olarak atar. |
| **Çalışma Şekli** | Veri setini bir dizi kurala göre bölerek ilerler (örn: `Sıcaklık > 25 C` mi?). En alttaki kutular (yapraklar), tahmin edilen sayısal değerdir (örn: 47.7 saat). |
| **✅ Artıları** | **Doğrusal olmayan ilişkileri** doğal olarak yönetir. Görselleştirmesi ve **yorumlaması kolaydır** (karar kurallarını takip edebilirsiniz). Veri ölçeklendirmeye (scaling) **ihtiyaç duymaz.** |
| **❌ Eksileri** | **Aşırı uyum (Overfitting)** riski yüksektir (çok derin ağaçlar). Verideki küçük değişiklikler, tüm ağaç yapısını değiştirebilir (kararsızdır). |

> **💡 Ana Çıkarım:** Karar Ağacı Regresyonu, veriyi bölgelere ayırır ve her bölge için **ortalama değeri** tahmin olarak kullanır. Bu sayede karmaşık ilişkiler modellenebilir, ancak **budama (pruning)** yapılmazsa aşırı uyum riski vardır.

---

## 🌲 Rastgele Orman Regresyonu (Random Forest Regression)

<img width="587" height="301" alt="image" src="https://github.com/user-attachments/assets/fb35efa7-8294-40c0-827f-06febae96e9d" />


| 🖼️ Özellik | Açıklama |
| :--- | :--- |
| **Model Tipi** | **Topluluk Yöntemi** (Ensemble Method). Birden çok Karar Ağacını birleştirir. |
| **Amacı** | Oluşturulan **çok sayıda Karar Ağacının tahminlerini ortalamak** (aggregation) yoluyla aşırı uyumu azaltmak ve performansı artırmak. |
| **Çalışma Şekli** | Rastgele seçilmiş özellikler ve veri alt kümeleri üzerinde yüzlerce farklı Karar Ağacı eğitilir. Test örneği her ağaçtan geçer ve sonuç olarak **tüm ağaçların tahminlerinin ortalaması** alınır. |
| **✅ Artıları** | Tek bir ağaca göre **daha sağlam (robust) ve daha doğrudur** (genellikle tek bir ağacın zayıflıklarını ortadan kaldırır). Karmaşık, doğrusal olmayan ilişkiler için mükemmeldir. Gürültüye (noise) karşı daha az hassastır. |
| **❌ Eksileri** | Tek bir ağaca göre **daha az yorumlanabilir.** Eğitimi ve tahmin yapması basit modellere göre **daha yavaştır.** |

> **💡 Ana Çıkarım:** Rastgele Orman, birden fazla ağacın tahminlerini birleştirerek **varyansı azaltır**, bu da onu tek bir ağaçtan daha doğru ve sağlam yapar. Çoğu zaman bireysel ağaçlardan daha iyi performans gösteren "tercih edilen" bir algoritmadır.


# 🏘️ Regresyon Modelleri: k-En Yakın Komşu (kNN) Regresyonu

<img width="704" height="308" alt="image" src="https://github.com/user-attachments/assets/74883a25-0af6-4baf-86bf-662a42299155" />


k-En Yakın Komşu (kNN) algoritması hem sınıflandırma hem de regresyon görevlerinde kullanılabilen, sezgisel bir makine öğrenimi modelidir.

---

## 5️⃣ k-En Yakın Komşu (kNN) Regresyonu

| 🖼️ Özellik | Açıklama |
| :--- | :--- |
| **Model Tipi** | **Parametrik olmayan**, örneğe dayalı öğrenme modelidir. Model, veriyi eğitme sırasında değil, yalnızca tahmin anında kullanır. |
| **Amacı** | Bir veri noktasının değerini, ona **en yakın olan $k$ komşunun hedef değerlerini ortalayarak** tahmin eder. |
| **Çalışma Şekli** | Yeni bir veri noktası geldiğinde, modele en yakın $k$ noktası bulunur. Bu $k$ komşunun gerçek sayısal değerleri toplanır ve ortalaması alınarak yeni noktanın tahmini yapılır. |
| **Örnek** | Tahmin edilen evin fiyatı, ona en yakın 3 evin fiyatının (14.2, 14.7, 15.3 - $100k$) ortalaması ($\approx 14.7$) olur. |
| **Gereklilik** | Tüm özelliklerin, komşuları bulurken eşit ağırlığa sahip olması için **ölçeklendirme (scaling)** esastır. |
| **✅ Artıları** | Çok **sezgiseldir** ve anlaşılması kolaydır. Veri dağılımı hakkında hiçbir varsayımı yoktur (doğrusal olmayan ilişkileri de yakalar). |
| **❌ Eksileri** | **Ölçeklendirmeye ve gürültüye (noise) karşı hassastır.** Büyük veri setlerinde tahmin yapması yavaştır (çünkü her tahmin için tüm veri setini tarar). |

---

## 🆚 kNN Sınıflandırma vs. kNN Regresyonu

| Görev | 🎯 Sınıflandırma (Classification) | 📉 Regresyon (Regression) |
| :--- | :--- | :--- |
| **Soru** | "Bu nokta **hangi sınıfa** aittir?" | "Bu nokta için **hangi sayısal değeri** tahmin etmeliyiz?" |
| **Tahmin Metodu** | $k$ komşunun **çoğunluk oyu** (Majority Vote). | $k$ komşunun hedef değerlerinin **ortalama değeri** (Average Value). |
| **Örnek** | En yakın 5 noktanın çoğu Yıldız ise, yeni nokta da Yıldız olarak etiketlenir. | En yakın 5 noktanın hedef değerleri 10, 12, 11, 13, 14 ise, tahmin $12$ olur. |

> **💡 Ana Çıkarım:** kNN Regresyonu, bir noktaya en yakın komşuların ortalama değerini kullanarak tahmin yapan, basit ve güçlü bir yöntemdir. Ancak etkin çalışması için **veri ölçeklendirmesi zorunludur** ve büyük veri setlerinde yavaşlama eğilimi gösterebilir.
