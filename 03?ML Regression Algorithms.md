
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


