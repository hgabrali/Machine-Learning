# 🧠 Makine Öğrenimi'nin Olasılık Temelleri (Probability Foundations of Machine Learning)

Makine öğrenimi, büyük ölçüde bilinmeyen bir sonucun, elimizdeki verilere dayanarak ortaya çıkma **olasılığını hesaplama** sanatıdır. Temelde ML, bir olasılık hesaplama makinesi olarak işlev görür.

---

## 1. Koşullu Olasılık (Conditional Probability) 📊

Makine öğrenimindeki en yaygın kullanım alanı, bir şeyin olasılığını **başka faktörler (features)** verildiğinde hesaplamaktır.

| Uygulama (Application) | Hesaplanan Koşullu Olasılık | Açıklama (Explanation) |
| :--- | :--- | :--- |
| **Spam Tespiti** (Spam Detection) 📧 | P(Spam \| Kelimeler, Alıcılar, ...) | Bir e-postanın **istenmeyen posta (spam)** olma olasılığı, e-postadaki **kelimeler (words)** ve diğer **özellikler (features)** verildiğinde. |
| **Duygu Analizi** (Sentiment Analysis) 😊 | P(Mutlu \| Metindeki Kelimeler) | Bir metnin **mutlu (happy)** olma olasılığı, içerdiği **kelimeler (words)** verildiğinde. |
| **Görüntü Tanıma** (Image Recognition) 📸 | P(Kedi \| Pikseller) | Bir resimde **kedi (cat)** olma olasılığı, resimdeki **pikseller (pixels)** verildiğinde. |

**Teknik Tanım:** Koşullu olasılık, **P(A \| B)** formülü ile ifade edilir ve "**B olayı gerçekleştiğinde A olayının gerçekleşme olasılığı**" anlamına gelir. Makine öğrenimi **sınıflandırıcılarının (classifiers)** yaptığı temel iş budur.

---

## 2. Süpervizyonlu Öğrenme ve Koşullu Olasılık (Supervised Learning and Conditional Probability)

Metinde bahsedilen Spam Tespiti, Duygu Analizi ve Görüntü Tanıma örnekleri **Süpervizyonlu Öğrenme (Supervised Learning)** alanına aittir.

* Bu alanda, model **etiketli veri (labeled data)** kullanılarak eğitilir ve sorulara cevap arar.
* **Amaç:** Verilen girdi için doğru **etiketi (label)** tahmin etmektir.
* **Örnekler:** Resim kedi mi? Cümle olumlu mu? E-posta spam mi?

---

## 3. Bayes Teoremi ve Artçı Olasılık (Bayes' Theorem and Posterior Probability) 🎲

Bayes Teoremi, bir **sınıflandırıcı (classifier)** oluşturarak bir şeyin olasılığını başka bir şey verildiğinde hesaplamanızı sağlayan matematiksel bir mekanizma sunar.

$$\mathbf{P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}}$$

Olasılığın hesaplanmasında **Bayes Teoremi'nin** nasıl kullanıldığı açıklanmıştır:

1.  **Önsel Olasılık (Prior Probability):** Başlangıç olasılığıdır (Örn: Tüm e-postalar içinde spam oranı, P(Spam)).
2.  **Kanıt (Evidence) veya Olay (Event):** Yeni bir bilginin ortaya çıkması (Örn: E-postada "lottery" kelimesinin geçmesi).
3.  **Artçı Olasılık (Posterior Probability):** Önsel olasılığın yeni bilgiyle güncellenmiş halidir (Örn: P(Spam \| Lottery)).

---

## 4. Jeneratif Modeller ve Saf Olasılık (Generative Models and Pure Probability) ✨

Metin, koşullu olasılığın yanı sıra **saf olasılıkların (pure probabilities)** kullanıldığı büyük bir ML alanını daha tanıtır: **Jeneratif Makine Öğrenimi (Generative Machine Learning)**. Bu, **Gözetimsiz Öğrenme (Unsupervised Learning)** alanının bir parçasıdır.

* **Amaç:** Modelin, var olan veriye benzeyen **yeni içerik** (görüntü, metin, ses) üretmesini sağlamaktır.
* **İşleyiş:** Burada amaç, koşullu olasılık hesaplamak yerine, yeni üretilen verinin gerçekçi olma olasılığını **maksimize etmektir (maximize probability)**.
    * **Görüntü Üretimi (Image Generation - Örn: StyleGAN):** Model, rastgele piksellerin insan yüzü oluşturma olasılığını **maksimize etmeye** çalışır.
    * **Metin Üretimi (Text Generation):** Model, rastgele kelimelerin anlamlı ve bağlamına uygun bir metin oluşturma olasılığını **maksimize etmeye** çalışır.
 
    * # 💡 Makine Öğrenmesinde Bayes Teoremi ve İlgili Konular

Bayes Teoremi, makine öğrenmesinde **olasılıksal sınıflandırma** algoritmalarının temelini oluşturur. Özellikle **Naïve Bayes Sınıflandırıcıları** gibi yaygın algoritmaları anlamak için bu kavramlara derinlemesine hakim olmak kritiktir.

---

## 📐 Bayes Teoremi'nin Formülü ve Bileşenleri

Bayes Teoremi, bir olayın gerçekleşme olasılığını, önceden bilinen koşullu ve marjinal olasılıkları kullanarak hesaplar.

$$\huge P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

| Terim | Adı | Açıklama (Makine Öğrenmesi) |
| :--- | :--- | :--- |
| **P(A|B)** | **Sonsal Olasılık (Posterior)** | Verilen veri (B) ile örneğin belli bir sınıfa (A) ait olma olasılığı. (Hedefimiz budur!) |
| **P(B|A)** | **Olabilirlik (Likelihood)** | Verilen sınıf (A) için verinin (B) gözlemlenme olasılığı. |
| **P(A)** | **Önsel Olasılık (Prior)** | B olayı hakkında hiçbir bilgi olmadan A sınıfının genel olasılığı. |
| **P(B)** | **Kanıt (Evidence)** | Sonsal olasılığı normalize eden sabittir. |

## 🤖 Naive Bayes Sınıflandırıcı Çeşitleri ve Karşılaştırması

Naive Bayes, **özelliklerin birbirinden bağımsız olduğu** (saf varsayım) varsayımına dayanır ve basitliği ile öne çıkar.

| Algoritma Adı | Temel Özellik Varsayımı | Uygulama Alanı | Örnek Veri Tipi | Avantajları |
| :--- | :--- | :--- | :--- | :--- |
| **Gaussian Naive Bayes** | Özellikler **Normal (Gauss) dağılımına** (Normal/Gaussian Distribution) uyar. | Sürekli sayısal verilerin olduğu sınıflandırma problemleri. | Boy, kilo, sıcaklık gibi sürekli değerler. | Hızlı ve küçük veri kümelerinde başarılıdır. |
| **Multinomial Naive Bayes** | Özellikler **Multinomial dağılımına** (Multinomial Distribution) uyar (sayım verileri). | Metin sınıflandırma (spam, duygu analizi). | Bir belgedeki kelimelerin frekansı (sayımı). | Metin verilerinde en başarılı çeşitlerdendir. |
| **Bernoulli Naive Bayes** | Özellikler **Bernoulli dağılımına** (Bernoulli Distribution) uyar (ikili/Boolean). | Belge varlığı/yokluğu, ikili özelliklerin olduğu sınıflandırmalar. | Kelimenin bir belgede bulunup bulunmaması (1/0). | İkili özellik setleri için etkilidir. |

---

## 🧠 Bilinmesi Gereken Kritik Ek Konular

Bayes algoritmalarını etkili kullanmak için anlaşılması gereken temel kavramlar:

### 1. Koşullu Bağımsızlık Varsayımı (Conditional Independence)
* **Açıklama:** Naïve Bayes'in "saf" kısmı buradan gelir. Hedef sınıf verildiğinde, girdilerin (özelliklerin) birbirlerinden bağımsız olduğu varsayılır. Bu, hesaplama karmaşıklığını önemli ölçüde azaltır.
* **Matematiksel İfade:** $P(x_1, x_2, \dots, x_n | C) = \prod_{i=1}^{n} P(x_i | C)$

### 2. Sıfır Frekans Problemi ve Çözümü
* **Problem:** Eğitim verisinde hiç görülmeyen bir özellik-sınıf kombinasyonu test verisinde ortaya çıkarsa, ilgili **Olabilirlik** ($P(B|A)$) sıfır olur. Bu durumda, sonuç **Sonsal Olasılık** da sıfırlanır.
* **Çözüm:** **Laplace Yumuşatması (Laplace Smoothing)** kullanılır. Tüm sayımlara küçük bir pozitif değer ($\alpha$, genellikle 1) eklenerek sıfır olasılıklar engellenir.

### 3. Maksimum Sonsal Olasılık (Maximum A Posteriori - MAP) Karar Kuralı

* **Açıklama:** Sınıflandırma yaparken Naïve Bayes, olası tüm sınıflar ($C_k$) arasından **en yüksek sonsal olasılığa** sahip olan sınıfı seçer. Bu, modelin tahmin mekanizmasıdır.


### 4. Bayes Ağları (Bayesian Networks) 🌐
* **Açıklama:** Naïve Bayes'in bağımsızlık varsayımının ötesine geçen, özellikler arasındaki **bağımlılıkları** modelleyebilen daha gelişmiş olasılıksal grafik modellerdir.
* **Önemi:** Özellikler arasındaki nedensel ilişkileri (Yönlendirilmiş Asiklik Grafikler - DAG) kullanarak daha doğru, ancak daha karmaşık çıkarımlar sağlar.

---

## ✉️ Örnek Uygulama: E-posta Spam Tespiti (Multinomial NB)

**Amaç:** "Ücretsiz para" e-postasını sınıflandırmak.

### 1. Önsel Olasılıklar (Prior)

| Sınıf | Sayım | $P(C)$ |
| :--- | :--- | :--- |
| Spam | 3 | $3/5 = 0.6$ |
| Ham | 2 | $2/5 = 0.4$ |

### 2. Olabilirlikler (Likelihood) - Laplace Yumuşatması Uygulanmış

Kullanılan kelimeler: "Ücretsiz" ve "Para".

* $P(\text{Ücretsiz}|\text{Spam}) = 3/17$
* $P(\text{Para}|\text{Spam}) = 2/17$

* $P(\text{Ücretsiz}|\text{Ham}) = 1/14$
* $P(\text{Para}|\text{Ham}) = 1/14$

### 3. Sonsal Olasılık Hesabı

MAP kuralını uygulayarak:

* **Spam Puanı:** $P(\text{Spam}|D) \propto P(\text{Ücretsiz}|\text{Spam}) \cdot P(\text{Para}|\text{Spam}) \cdot P(\text{Spam})$
    * Spam Puanı $\propto (3/17) \cdot (2/17) \cdot 0.6 \approx 0.0124$

* **Ham Puanı:** $P(\text{Ham}|D) \propto P(\text{Ücretsiz}|\text{Ham}) \cdot P(\text{Para}|\text{Ham}) \cdot P(\text{Ham})$
    * Ham Puanı $\propto (1/14) \cdot (1/14) \cdot 0.4 \approx 0.0020$

**Sonuç:** $0.0124 > 0.0020$ olduğu için model, e-postayı **SPAM** olarak sınıflandırır. ✅

---
---

<img width="930" height="347" alt="image" src="https://github.com/user-attachments/assets/ebf529b1-ea70-4f52-b2cc-874a6a8607f4" />

## ❓ Olasılık Problemi Çözümü: Yetersiz Bilgi

Bu, **Birleşim Olasılığı (Probability of Union)** hesaplama problemidir. Temel kural, iki olayın **kesişiminin ($P(A \text{ ve } B)$)** bilinmesini gerektirir.

### 💊 Deney Verileri

* **Toplam Hasta Sayısı:** 100
* **Baş Ağrısı Yaşayanlar (Headache - H):** 50
* **Ateş Yaşayanlar (Fever - F):** 50

**Olasılıklar (Prior):**
* $P(H) = \frac{50}{100} = 0.5$
* $P(F) = \frac{50}{100} = 0.5$

### 🎯 İstenen Olasılık

Doktorlar, bir hastanın **Baş Ağrısı VEYA Ateş** yaşama olasılığını, yani $\mathbf{P(H \cup F)}$'i bulmak istiyor.

### 📝 Olasılıkların Birleşim Kuralı (The Addition Rule)

Genel kural şudur:
$$\mathbf{P(A \text{ veya } B)} = P(A) + P(B) - P(A \text{ ve } B)$$

Bu kuralı uygulamak için $\mathbf{P(\text{Baş Ağrısı ve Ateş})}$ terimine ihtiyacımız var. Bu, aynı anda hem baş ağrısı hem de ateş yaşayan hastaların oranını ifade eder. Soruda bu **kesişim olasılığı** (yani kaç hastanın çifte semptom yaşadığı) **verilmemiştir**.

### ❌ Geçersiz Varsayımların İncelenmesi

| Varsayım | Kural | Neden Geçersiz? |
| :--- | :--- | :--- |
| **Ayrık Olaylar** (Mutually Exclusive) | $P(A \text{ veya } B) = P(A) + P(B) = 1$ | Hiçbir hastanın iki semptomu birden yaşamadığı varsayılır. Soruda bu bilgi **yoktur**. |
| **Bağımsız Olaylar** (Independent) | $P(A \text{ ve } B) = P(A) \cdot P(B) = 0.25$ | Tıbbi semptomlar genellikle **bağımlıdır**. İlaç, semptomlardan birini tetikleyebilir veya engelleyebilir. Bu varsayımın doğru olduğu **garanti edilemez**. |

### 🛑 Sonuç

$\mathbf{P(\text{H ve F})}$ değeri (kesişim) bilinmediği sürece, **Birleşim Olasılığı** doğru bir şekilde hesaplanamaz.

* **Ek Bilgi Gereksinimi:** Kaç hasta **sadece** baş ağrısı, kaç hasta **sadece** ateş ve kaç hasta **hem** baş ağrısı **hem de** ateş yaşadı?

**Doğru İfade:**
> **Not enough information is given to calculate $P(\text{fever or headache})$.**
> (Baş ağrısı veya ateş olasılığını hesaplamak için yeterli bilgi verilmemiştir.)
>
> ## 📌 Açıklamanın Özeti: Neden Yetersiz Bilgi?

Açıklama, temel olasılık kurallarından yola çıkarak şunları söylüyor:

1.  **Olaylar Ortaktır (Joint Events) 🤝:** Problem metninde, iki olayın (**baş ağrısı** ve **ateş**) **ayrık (disjoint)** olduğu, yani aynı anda gerçekleşemeyeceği söylenmiyor. Dolayısıyla, bazı kişilerin **hem baş ağrısı hem de ateş** yaşama ihtimali vardır. Bu tür çakışan olaylara **ortak olaylar (joint events)** denir.

2.  **Kesişim Bilinmeli ❓:** İki olayın birleşim olasılığını ($P(A \text{ veya } B)$) hesaplamak için genel **Toplama Kuralı (Addition Rule)**'nı kullanmak zorunludur:
    $$\mathbf{P(A \text{ veya } B)} = P(A) + P(B) - P(A \text{ ve } B)$$

3.  **Sonuç 🛑:** Bu nedenle, $P(\text{ateş veya baş ağrısı})$ olasılığını bulmak için, olayların kesişim olasılığı olan **$P(\text{ateş VE baş ağrısı})$** değerini **bilmeniz gerekir**. Bu bilgi problem metninde verilmediği için, olasılık hesaplanamaz.

---

<img width="870" height="357" alt="image" src="https://github.com/user-attachments/assets/f76afc36-6297-45df-9f2a-011289ddd967" />


## 🐞 Koşullu Olasılık Çözümü: Yazılım Testi

Bu problem, **Bayes Teoremi'nin temelini oluşturan** bir **Koşullu Olasılık (Conditional Probability)** problemidir. Bir kullanıcının hata deneyimlediği bilgisi **verildiğinde**, Versiyon B'yi test etme olasılığını bulmayı amaçlar.

### 📝 1. Olay Tanımları ve Formül

* **B:** Kullanıcının **Versiyon B**'yi test etmesi.
* **H:** Kullanıcının **Hata (Bug)** deneyimlemesi.

Aranan olasılık: $P(B|H)$ (Hata deneyimlediğine göre, Versiyon B'yi test etme olasılığı).

$$\mathbf{P(B|H)} = \frac{P(B \cap H)}{P(H)}$$

### 📊 2. Verilerin Özetlenmesi

Toplam kullanıcı sayısı: $4000 + 5000 = \mathbf{9000}$

| Kategori | Değer | Olay Sayısı (N) | Olasılık (P) |
| :--- | :--- | :--- | :--- |
| **Versiyon B Kullanıcıları** | 5000 | $N(B)$ | $P(B) = 5000/9000$ |
| **Toplam Hata Deneyimleyenler** | 3000 | $N(H)$ | $P(H) = 3000/9000$ |
| **Versiyon B ve Hata Kesişimi** | 1500 | $N(B \cap H)$ | $P(B \cap H) = 1500/9000$ |

### 🧠 3. Hesaplama (Kullanıcı Sayılarıyla)

Olasılıklar yerine, hesaplamanın daha basit olması için doğrudan **kullanıcı sayılarını** kullanabiliriz, çünkü payda ($N(\text{Toplam})$) sadeleşecektir:

$$\mathbf{P(B|H)} = \frac{N(\text{B ve H})}{N(H)} = \frac{\text{Versiyon B ile hata deneyimleyenler}}{\text{Toplam hata deneyimleyenler}}$$

$$P(B|H) = \frac{1500}{3000}$$

$$P(B|H) = \frac{1}{2} = \mathbf{0.5}$$

---

### ✅ Sonuç

Bir kullanıcının **hata deneyimlediği** bilgisi verildiğinde, bu kullanıcının **Versiyon B**'yi test etmiş olma olasılığı $\mathbf{1/2}$ (**%50**) olarak bulunur.
