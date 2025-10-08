# Introduction to Machine Learning

<img width="451" height="463" alt="image" src="https://github.com/user-attachments/assets/3f6b03fa-9578-4f27-b01a-db56cfa80412" />

## ☝🏼 In summary:
* AI → systems that mimic human intelligence.
* ML → algorithms that learn from data.
* Deep Learning → ML powered by neural networks.
* LLMs → models like ChatGPT that understand and generate text.
* GenAI → AI that creates new content (text, images, code, etc.).

 [AI, Machine Learning, Deep Learning and Generative AI Explained](https://www.youtube.com/watch?v=qYNweeDHiyU)

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
