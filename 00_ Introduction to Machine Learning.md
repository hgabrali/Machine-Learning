# Introduction to Machine Learning

<img width="451" height="463" alt="image" src="https://github.com/user-attachments/assets/3f6b03fa-9578-4f27-b01a-db56cfa80412" />

## ☝🏼 In summary:
* AI → systems that mimic human intelligence.
* ML → algorithms that learn from data.
* Deep Learning → ML powered by neural networks.
* LLMs → models like ChatGPT that understand and generate text.
* GenAI → AI that creates new content (text, images, code, etc.).

 [AI, Machine Learning, Deep Learning and Generative AI Explained](https://www.youtube.com/watch?v=qYNweeDHiyU)


 <img width="1179" height="665" alt="image" src="https://github.com/user-attachments/assets/ebec45ef-fa9b-4b7b-ac9a-529308cb9546" />
 
* [Picture](https://blogs.sas.com/content/subconsciousmusings/2020/12/09/machine-learning-algorithm-use/)

# 🤖  What is Machine Learning 🧠
Machine Learning (ML) is a branch of Artificial Intelligence (AI) that teaches computers to learn from data and improve over time—without being explicitly programmed.


## 💻 Types of Machine Learning

## 🆚 Supervised vs. Unsupervised Learning: Güncel Sektör Örnekleri

| Özellik | Supervised Learning (Gözetimli Öğrenme) 🧑‍🏫 | Unsupervised Learning (Gözetimsiz Öğrenme) 🧩 |
| :--- | :--- | :--- |
| **Tanım** | Algoritma, **etiketli (labeled)** veri setinden öğrenir ve girdi ile çıktı arasındaki ilişkiyi eşleştirir. | Algoritma, **etiketsiz (unlabeled)** verideki gizli yapıları, desenleri ve benzerlikleri bulur. |
| **Amaç** | Doğrulukla **tahmin (prediction)** yapmak veya **sınıflandırma (classification)** yapmak. | Veriyi **keşfetmek (exploration)**, organize etmek ve gruplamak. |
| **Yaygın Algoritmalar** | Lineer Regresyon, Lojistik Regresyon, Destek Vektör Makineleri (SVM), Karar Ağaçları/Rastgele Orman. | K-Means Kümeleme (Clustering), Hiyerarşik Kümeleme, Temel Bileşen Analizi (PCA). |
| **Sektörel Örnek 1: Finans** 💰 | Bir müşterinin kredi notuna göre **kredi batık riskini** (Evet/Hayır) tahmin etmek. | Bir bankanın müşteri harcama davranışlarına göre **farklı müşteri segmentleri** (örneğin, "Yüksek Risk", "Normal", "Düşük Risk") oluşturması. |
| **Sektörel Örnek 2: Sağlık** 🩺 | Hasta verilerine dayanarak bir tümörün **iyi huylu mu yoksa kötü huylu mu** olduğunu sınıflandırmak. | Bilinen bir rahatsızlığı olmayan hastalardaki genetik verilere bakarak **yeni bir hastalık alt türünü** keşfetmek. |
| **Sektörel Örnek 3: E-Ticaret** 🛍️ | Bir e-posta'nın **spam olup olmadığını** sınıflandırmak. | Kullanıcıların web sitesinde yaptığı tıklama ve gezinti yollarına göre **ürünleri otomatik olarak kategorilere** ayırmak (ürünlerin etiketleri manuel verilmeden). |
| **Sektörel Örnek 4: NLP** 💬 | Bir müşteri yorumunun **olumlu veya olumsuz** duygu içerdiğini belirlemek (**Sentiment Analysis**). | Bir doküman koleksiyonundaki kelime kullanım sıklıklarına bakarak **ana konuları** (örneğin, "Politika", "Spor", "Ekonomi") otomatik olarak belirlemek (**Topic Modeling**). |

☝🏼☝🏼☝🏼☝🏼☝🏼
* Supervised learning is often more accurate because we give the computer answers to learn from.
* But labeling data can be expensive and slow. That’s why unsupervised learning is also widely used—it can find insights in raw, unlabeled data.

* ## 💻 Yaygın ML Algoritmaları: İşlevlerine Göre Karşılaştırma

  <img width="1002" height="512" alt="image" src="https://github.com/user-attachments/assets/c03d588c-ebfb-4d14-889e-e17514740a46" />
  <img width="984" height="515" alt="image" src="https://github.com/user-attachments/assets/768b112b-0a59-41a6-9552-4aa6f6c05931" />
  <img width="663" height="665" alt="image" src="https://github.com/user-attachments/assets/c446b707-941b-44a1-b380-8500565b7968" />




| Algoritma Tipi | Algoritma (Örnekler) | Öğrenme Türü | Temel İşlevi ve Ne İşe Yarar? |
| :--- | :--- | :--- | :--- |
| **Supervised** 🧑‍🏫 | **Lojistik Regresyon** | Sınıflandırma (Classification) | İkili (binary) sonuçları tahmin etmek (Evet/Hayır). Örneğin, bir e-postanın spam olup olmadığını tahmin eder. |
| **Supervised** 🎯 | **Karar Ağaçları / Rastgele Orman** | Sınıflandırma & Regresyon | Veriyi kararlar zinciri (ağacı) ile ayırarak tahmin yapar. Karmaşık, doğrusal olmayan ilişkileri modellemek için güçlüdür. |
| **Supervised** 📈 | **Lineer Regresyon** | Regresyon (Regression) | Girdiler ile sürekli bir hedef değişken arasındaki doğrusal ilişkiyi modelleyerek tahmin yapar. Örneğin, metrekareye göre ev fiyatını tahmin etmek. |
| **Supervised** 📐 | **Destek Vektör Makineleri (SVM)** | Sınıflandırma & Regresyon | İki sınıfı birbirinden ayıran en iyi sınırı (hiper düzlemi) bularak sınıflandırma yapar. Yüksek boyutlu verilerde etkilidir. |
| **Unsupervised** 🧩 | **K-Means Kümeleme** | Kümeleme (Clustering) | Etiketsiz verileri, benzerliklerine göre belirlenen K adet gruba (kümeye) ayırır. Örneğin, pazarlama için müşteri segmentasyonu yapmak. |
| **Unsupervised** ⬇️ | **Temel Bileşen Analizi (PCA)** | Boyut Azaltma (Dimensionality Reduction) | Veri setindeki gürültüyü azaltmak ve temel bilgiyi koruyarak değişken sayısını düşürmek için kullanılır. |
| **Unsupervised** 🔗 | **Hiyerarşik Kümeleme** | Kümeleme (Clustering) | Veri noktaları arasında bir hiyerarşi veya ağaç yapısı (dendrogram) oluşturarak kümeler oluşturur. |



## 🎯 ML Kullanım Senaryoları: Analiz ve Karşılaştırma

Aşağıdaki tablo, görselde verilen örneklerin hangi ML türüyle çözüldüğünü ve neden o türün kullanılması gerektiğini açıklamaktadır.

| İş Senaryosu (Girdi) | ML Türü | Temel ML Görevi | Açıklama / Gerekçe |
| :--- | :--- | :--- | :--- |
| **Predicting the winner of the next football match** ⚽ (Bir sonraki futbol maçının galibini tahmin etme) | **Supervised Learning** (Gözetimli Öğrenme) 🧑‍🏫 | Classification (Sınıflandırma) | Sonuç **kategoriktir** (Kazan/Kaybet/Berabere). Geçmiş etiketli verilere (maç sonuçları) dayanarak bir kategori (sınıf) tahmin edilir. |
| **Discover hidden sub-groups within your existing customers** 🕵️ (Mevcut müşterileriniz arasında gizli alt grupları keşfetme) | **Unsupervised Learning** (Gözetimsiz Öğrenme) 🧩 | Clustering (Kümeleme) | Veri **etiketsizdir**. Algoritma, herhangi bir ön bilgi (label) olmadan müşteriler arasındaki doğal benzerlikleri bulur ve gruplar. |
| **Predict how much revenue a new lead will generate in their first year** 💰 (Yeni bir potansiyel müşterinin ilk yılında ne kadar gelir getireceğini tahmin etme) | **Supervised Learning** (Gözetimli Öğrenme) 📈 | Regression (Regresyon) | Sonuç **sürekli bir sayıdır** (gelir miktarı). Sayısal bir değer tahmin edildiği için regresyon kullanılır. |
| **Grouping customers into segments based on shopping behavior** 🛍️ (Alışveriş davranışına göre müşterileri segmentlere ayırma) | **Unsupervised Learning** (Gözetimsiz Öğrenme) 🔗 | Clustering (Kümeleme) | Amaç, benzer davranış sergileyen müşterileri **etiket bilgisi olmadan** gruplamaktır. Bu, keşif amaçlı bir görevdir. |
| **Predict which new leads are most likely to become paying customers** ✅ (Hangi yeni potansiyel müşterilerin ödeme yapan müşteriye dönüşme olasılığının en yüksek olduğunu tahmin etme) | **Supervised Learning** 🎯 | Classification (Sınıflandırma) | Sonuç **kategoriktir** (Evet/Hayır). Potansiyel müşteri, ödeme yapacak (Evet) veya yapmayacak (Hayır) olarak sınıflandırılır. |


# How Machine Learning Works: Overview of a DS Lifecycle

<img width="616" height="616" alt="image" src="https://github.com/user-attachments/assets/c28b5887-6412-4a29-a2ef-6e49c7b9787e" />


[Data Science Lifecycle in more detail](https://www.youtube.com/watch?v=4Cp6PkBKqX4)

## 🧠 How Machine Learning Works: Overview of a DS Lifecycle

Makine Öğrenimi (ML) çözümü geliştirmek, sadece bir algoritma çalıştırmaktan ibaret değildir; bu, iş problemiyle başlayan ve sürekli takiple biten yedi aşamalı döngüsel bir süreçtir.

| Aşama No. | Aşama Adı | Görev Tanımı ve Amacı | Örnek Uygulama (Müşteri Kaybı Çözümünde) |
| :---: | :--- | :--- | :--- |
| **1** ❓ | **Problem Definition** | **Net bir soru** belirleyerek projenin amacını tanımlar. Başarılı bir ML projesi için gerekli ilk adımdır. | **Soru:** "Müşteriler bizi neden terk ediyor ve tam olarak neyi tahmin etmek istiyoruz?" |
| **2** 🧺 | **Data Collection** | Tanımlanan problem için gerekli olan **ilgili verileri** (çeşitli kaynaklardan) toplar. | İşlem günlükleri, müşteri profilleri ve destek hattı verilerini toplamak. |
| **3** 🧹 | **Data Preparation** | Ham veriler dağınıktır; bu aşamada veriler **temizlenir, organize edilir** ve ML modelinin kullanabileceği formata dönüştürülür. | Eksik değerleri temizlemek, çift kayıtları (duplicate) kaldırmak ve "müşteri bekleme süresi" gibi yeni özellikler oluşturmak. |
| **4** 🔎 | **Data Exploration** | Model kurulmadan önce veriyi anlamak, **kalıpları, eğilimleri** ve aykırı değerleri (outlier) görselleştirmeler ve istatistiklerle keşfetmek. | Müşterilerin ayda 3 kereden fazla destek birimiyle iletişime geçtiğinde ayrılma olasılığının daha yüksek olduğunu keşfetmek. |
| **5** 🧠 | **Model Building** | Uygun algoritma seçilir ve tarihsel (eğitim) verileri kullanılarak **model eğitilir**. | **Lojistik Regresyon** gibi bir model seçmek ve bu modeli tarihsel kayıp (churn) verileriyle eğitmek. |
| **6** ⚖️ | **Model Evaluation** | Modelin **görülmemiş test verileri** üzerindeki performansını kontrol eder. Performans düşüklüğünde, model hiper parametreleri ayarlanır veya başka algoritmalar denenir. | Doğruluk (**accuracy**) ve geri çağırma (**recall**) metrikleriyle performansı ölçmek, ardından hiper parametreleri ayarlayarak sonuçları iyileştirmek. |
| **7** 🚀 | **Model Deployment & Monitoring** | Model gerçek zamanlı tahminler yapabilmesi için üretim ortamına alınır (**deploy** edilir). İş bitmemiştir; modelin zamanla değişen veri kalıplarına uyum sağlaması için **izlenmesi ve gerektiğinde yeniden eğitilmesi** gerekir. | Modeli üretime almak, böylece satış ekibinin anlık kayıp riski puanlarını görmesini sağlamak ve kayıp kalıpları değiştikçe modeli izleyip yeniden eğitmek. |

## ✅ Müşteri Kaybı (Churn) Senaryosu İçin Doğru Sıralama

İlk görseldeki adımlar, yukarıdaki yaşam döngüsü aşamalarına göre sıralanmıştır.

| Adım No. | Açıklama | Karşılığı Olan Aşama |
| :---: | :--- | :--- |
| **4** ❓ | Ask: "Why are customers leaving us, and what exactly do we want to predict?" | **1. Problem Definition** |
| **3** 📊 | Collect transaction logs, customer profiles, and support call data. | **2. Data Collection** |
| **1** 🧹 | Clean missing values, remove duplicates, and create new features... | **3. Data Preparation** |
| **6** 🧠 | Choose logistic regression as a model and train it on historical data. | **5. Model Building** |
| **5** ⚖️ | Measure performance with accuracy and recall, then tweak hyperparameters... | **6. Model Evaluation** |
| **2** 🚀 | Deploy the model into production so the sales team can see churn-risk scores... | **7. Model Deployment** |
| **7** 🛡️ | Monitor the model as churn patterns change over time and retrain when needed. | **7. Model Monitoring** |

**Doğru Yaşam Döngüsü Sıralaması:** **4 → 3 → 1 → 6 → 5 → 2 → 7**

---
# 📊 Makine Öğreniminde Temel Veri Türleri (Key Data Types in Machine Learning)

<img width="645" height="325" alt="image" src="https://github.com/user-attachments/assets/51b4a160-d9e4-4aad-85b3-07c8771f7518" />


<img width="682" height="676" alt="image" src="https://github.com/user-attachments/assets/5bda3bed-5ee4-4832-bbe4-186c6f124bed" />

<img width="911" height="424" alt="image" src="https://github.com/user-attachments/assets/62409c9e-277b-4f22-90b3-cda6fc8bc4c8" />


## Why Data Types are important?





### 🛠️ ML Veri Türleri ve Ön İşleme Gereksinimleri

**Veri Hazırlama (Data Preparation)**, Makine Öğreniminde (ML) çok kritik bir adımdır. Ham veri genellikle dağınık, eksik veya tutarsızdır ve veri temizlenmeden etkili bir model kurulamaz. Her bir veri türü, modelin onu anlaması için farklı ön işleme metotları (preprocessing methods) gerektirir.

| Ana Veri Tipi | Alt Tip | Örnekler | Neden Önemli? (Gerekçe) | Temel Ön İşleme Metotları |
| :--- | :--- | :--- | :--- | :--- |
| **Niceliksel** (Quantitative) 🔢 | **Kesikli** (Discrete) | 5 çocuk, 96 işçi, 3 dizüstü bilgisayar (Sayılabilir tam sayılar). | Model, büyüklükleri farklı olan sayıları orantısızca önemli görebilir. Ölçeklendirmenin yapılmaması model performansı düşer. | **Ölçeklendirme (Scaling):** Normalizasyon (Normalization) veya Standartizasyon (Standardization). |
| | **Sürekli** (Continuous) | 3.25 kg, 1.32 mil, 7.25 inç (Aralık içinde herhangi bir değeri alabilen sayılar). | Sürekli veriler genellikle büyük bir değer aralığına sahiptir. Bu, farklı özelliklerin model üzerindeki etkisinin dengelenmesi gerektiği anlamına gelir. | **Ölçeklendirme (Scaling):** Normalizasyon veya Standartizasyon. Eksik değerler için Ortalama/Medyan ile doldurma. |
| **Niteliksel** (Qualitative/Categorical) 🏷️ | **Nominal** (Nominal) | Kalem, Silgi, İnek, Köpek (Sırasız kategoriler). | **Bilgisayarlar kelimeleri anlamaz.** Bu kategorik etiketlerin, modelin işleyebileceği sayısal formata çevrilmesi gerekir. Sıralama önemsizdir. | **One-Hot Encoding** (Her kategori için yeni bir ikili [binary] sütun oluşturma). |
| | **Sırasal** (Ordinal) | Mükemmel, İyi, Kötü; Fantastik, Tamam, Beğenmedim (Sıralı kategoriler). | Sayısala çevrilirken, etiketler arasındaki **sıralamanın veya rütbenin korunması** çok önemlidir. | **Label Encoding** (Sıraya göre 1, 2, 3 gibi sayısal değerler atama). |


## 🛠️ Veri Hazırlamanın Genel Adımları (Data Preparation Steps)

<img width="1117" height="498" alt="image" src="https://github.com/user-attachments/assets/34ec9365-9998-47b4-bbce-a8042302ca8e" />

<img width="599" height="378" alt="image" src="https://github.com/user-attachments/assets/7886dd38-1076-4490-a15e-88f868fd165b" />


Bu adımlar, Makine Öğrenimi (ML) projesinin başlangıcında yapılır ve modelin öğrenme verimliliğini doğrudan etkiler.

1.  **Eksik Değerleri Yönetme (Handling missing values) ❓:** Boş hücreleri doldurma (imputation) veya kaldırma.
2.  **Kopya ve Hataları Kaldırma (Removing duplicates and errors) ❌:** Veri setinin temizliğini sağlama.
3.  **Kategorik Değişkenleri Kodlama (Encoding categorical variables) 🏷️:** Kategorik verileri sayısal formata çevirme.
4.  **Sayısal Değerleri Ölçeklendirme (Scaling or normalizing numerical values) ⚖️:** Sayısal özelliklerin model üzerindeki etkisini dengeleme.
5.  **Veriyi Ayırma (Splitting data) 🪓:** Veriyi eğitim (training) ve test (test) setlerine bölme.

> **Unutmayın:** Başarılı bir ML projesinin temeli, daima **temizlenmiş ve doğru şekilde hazırlanmış** veriye dayanır.


☝🏼 Key Idea: Knowing your data type is the first step in data preparation. It tells you which preprocessing methods and which algorithms can be applied.  

## 📊 Makine Öğreniminde Temel Veri Türleri (Key Data Types in Machine Learning)


**Ana Fikir:** Veri türünü bilmek, veri hazırlığının (data preparation) ilk adımıdır; hangi ön işleme yöntemlerini (preprocessing methods) ve hangi algoritmaların uygulanacağını belirler.

| Veri Tipi | Alt Tip | Açıklama (İngilizce Terim) | Önemi (Why it Matters) | Örnek Senaryo |
| :--- | :--- | :--- | :--- | :--- |
| **Sayısal** (Numeric / Quantitative) 🔢 | **Sürekli** (Continuous) | Belli bir aralıkta herhangi bir reel değeri alabilir (Can take any real number value within a range). | Sayıların ölçekleri çok farklı olabilir (örneğin maaş binlerle ifade edilirken, yaş onlarla ifade edilir). Model, daha büyük sayıları "daha önemli" olarak algılayabilir, bu yüzden ayar (scaling) gereklidir. | Kilo (Weight), Boy (Height) veya Sıcaklık (Temperature) değerleri. |
| | **Kesikli** (Discrete) | Sayılabilir miktarları temsil eder ve yalnızca tam sayı değerleri alır (Represents countable quantities and only takes integer values). | Sayısal verilerle aynı ölçek sorunlarına sahip olabilir. | Satılan ürün sayısı (The number of products sold) veya bir mağazadaki müşteri sayısı. |
| **Kategorik** (Categorical / Qualitative) 🏷️ | **Nominal** (Nominal) | Sıralanmamış kategori veya etiketlerdir; aralarında anlamlı bir sıra yoktur (Unordered categories). | Bilgisayarlar kelimeleri anlamaz. Bu kelimelerin sayılara dönüştürülmesi gerekir, ancak sıralama önemli olmadığı için her etikete rastgele bir sayı atanabilir. | Cinsiyet (Gender: male, female), Renkler (Colors: red, blue), veya ürün tipleri. |
| | **Sırasal** (Ordinal) | Anlamlı bir rütbe veya sıraya sahip sıralanmış kategorilerdir (Ordered categories with a meaningful rank or order). | Kelimeler sayılara dönüştürülürken **sıralama korunmalıdır** (örneğin: Lise < Lisans < Yüksek Lisans). | Müşteri memnuniyet anketi cevapları: Kötü (poor), Orta (fair), İyi (good), Mükemmel (excellent). |
| **Metin** (Text Data) 📝 | **Yapılandırılmamış** (Unstructured) | İncelemeler, tweet'ler veya e-postalar gibi yapılandırılmamış verilerdir (Unstructured words and sentences). | Bilgisayarlar metni sadece karakter dizileri olarak görür. Metinden öğrenmek için, metni sayılarla (örneğin, kelime sayarak) temsil etmemiz gerekir. | IMDb'den film incelemeleri listesi (A list of movie reviews from IMDb). |
| **Zaman Serisi** (Time Series Data) ⌚ | **Sürekli** (Continuous) | Düzenli aralıklarla (günlük, aylık vb.) toplanan verilerdir (Data collected at regular intervals). | Sıradan verilerin aksine, **sıralama önemlidir**. Dünün değeri bugünü etkiler ve iyi tahminler yapmak için trendler veya mevsimsellik yakalanmalıdır. | Apple hisse senedinin günlük kapanış fiyatı (Daily closing price of Apple stock). |
| **İkili** (Boolean / Binary Data) ✔️ | **İkili** (Binary) | Genellikle 0/1 veya Doğru/Yanlış olan iki olası değere sahiptir (Two possible values, often 0/1 or True/False). | İkili özellikler zaten sayısal olsa da, bir sınıf nadir ise (örneğin, e-postaların sadece %2'si spam ise) modelin her zaman "spam değil" tahmin ederek hile yapmaması için dengeleme (balancing) gerekebilir. | Bir müşterinin reklama tıklayıp tıklamadığını gösteren bir sütun ("1" tıklandı, "0" tıklanmadı). |
| **Görüntü** (Image Data) 🖼️ | **Yoğun** (Dense) | Gri tonlamalı (grayscale) veya RGB'den oluşan piksellerden yapılmıştır (Made up of pixels). | Tek bir görüntü **milyonlarca sayıya (piksele)** karşılık gelebilir. Modelin aşırı yüklenmemesi için görüntülerin işlenmesi özel teknikler gerektirir. | Bir hastanın beyninin MRI taraması (MRI scan of a patient's brain). |


## 🔢 Temel Veri Türlerine Somut Örnekler

Aşağıdaki tablo, farklı veri türlerinin gerçek hayattaki karşılıklarını göstermektedir.

| İş Senaryosu | Veri Türü (ML Kategorisi) | Açıklama |
| :--- | :--- | :--- |
| **Customer satisfaction survey** with answers: poor, fair, good, excellent. | **Categorical (Ordinal)** 🏷️ | Sıralanmış kategoriler (ordered categories). |
| **Daily closing price of Apple stock.** | **Time Series (Numeric, Continuous)** ⌚ | Zaman içinde takip edilen sayılar (numbers tracked over time). |
| **MRI scan of a patient's brain.** | **Image Data** 🖼️ | Piksel değerleri (pixel values). |
| **A column showing "1" if a customer clicked an ad, "0" if not.** | **Boolean (Binary)** ✔️ | Evet/Hayır sonucu (yes/no outcome). |
| **A list of movie reviews from IMDb.** | **Text Data** 📝 | Yapılandırılmamış kelimeler ve cümleler (unstructured words and sentences). |
