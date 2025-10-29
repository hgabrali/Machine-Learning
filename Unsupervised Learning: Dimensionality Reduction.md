# 📉 Boyut Azaltmaya (Dimensionality Reduction) Giriş

<img width="923" height="466" alt="image" src="https://github.com/user-attachments/assets/544aa4de-d05c-4919-b13d-4ee83bfd99b5" />

<img width="948" height="476" alt="image" src="https://github.com/user-attachments/assets/6baadaaf-29c2-4274-8914-7dbf68f23b5c" />


[Picture](https://www.geeksforgeeks.org/machine-learning/dimensionality-reduction/)


Boyut azaltma, veri setlerindeki gereksiz karmaşıklığı gidererek makine öğrenmesi modellerini daha verimli ve güçlü hale getiren kritik bir süreçtir.

## 1. Boyut Azaltma Nedir?

Çok sayıda özelliğe sahip veri kümelerine **yüksek boyutlu veri (high-dimensional data)** denir. Aşırı özellik sayısı, algoritmaların öğrenmesini zorlaştırabilir (Boyutların Laneti - The Curse of Dimensionality).

👉 **Boyut Azaltma:**
* Bir veri kümesindeki **önemli bilginin çoğunu koruyarak** özelliği daha az özellikle temsil etme eylemidir.

---

## 2. Neden İhtiyaç Duyarız? (Faydaları)

| Fayda | Açıklama |
| :--- | :--- |
| ✅ **Daha Hızlı Hesaplama** | Daha az özellik, modelleri eğitmek için **daha az zaman** demektir. |
| ✅ **Daha İyi Görselleştirme** | İndirgenmiş veriyi 2D veya 3D olarak çizerek **kümeleri (clusters)** veya paternleri "görebiliriz". |
| ✅ **Aşırı Öğrenmeden Kaçınma** | Daha az gürültülü özellik, modellerin yeni verilere **daha iyi genelleme yapmasını** sağlar. |
| ✅ **Fazlalığı Giderme** | Birbiriyle ilişkili (**correlated**) özellikleri alarak, onları daha az sayıda **bağımsız boyuta** sıkıştırır. |

---

## 3. Boyut Azaltma Tekniklerinin Türleri

| Kategori | Tanım | Örnek Algoritma |
| :--- | :--- | :--- |
| **🔹 Özellik Seçimi (Feature Selection)** | Yalnızca en alakalı özellikleri seçip geri kalanları atmak. | Rastgele Orman'dan (Random Forest) en önemli özellikleri tutmak. |
| **🔹 Özellik Çıkarma (Feature Extraction)** | Orijinal özelliklerden, daha düşük bir uzayda **yeni özellikler** oluşturmak. | **Temel Bileşen Analizi (PCA)** ve **t-SNE**. |

---

## 4. Zorluklar ve Sınırlamalar

| Sınırlama | Açıklama |
| :--- | :--- |
| ❌ **Yorumlanabilirlik Kaybı** | PCA ile oluşturulan yeni bileşenlerin anlamını açıklamak zordur. |
| ❌ **Bilgi Kaybı** | Süreç **kayıplıdır (lossy)**; bir miktar bilgi her zaman atılır. |
| ❌ **Deneysel Seçim** | İhtiyaç duyulan doğru boyut sayısını seçmek **açıklanan varyansa** (explained variance) bakarak deney gerektirir. |

---

## ☝🏽 Özet

* **Boyut Azaltma**, veri setlerini özellik sayısını azaltarak basitleştirir.
* **Özellik Seçimi** = Önemli olanları tutar.
* **Özellik Çıkarma** (PCA, t-SNE) = Yeni özellikler oluşturur.
* **Fayda:** Daha hızlı, daha az gürültü, daha iyi görselleştirme.
* **Denge:** Yorumlanabilirlik veya bir miktar veri bilgisi kaybolabilir.


