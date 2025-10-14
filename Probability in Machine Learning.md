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


---

<img width="966" height="326" alt="image" src="https://github.com/user-attachments/assets/c8262fec-89f0-4c5b-ac2f-3c45b1eb903f" />

# 🔬 Bayes Teoremi Uygulaması: Tıbbi Test Analizi

Bu, bir kişinin test sonucu pozitif çıktığında **gerçekte hasta olma olasılığını (Sonsal Olasılık)** hesaplayan klasik bir Bayes Teoremi problemidir.

### 🎯 Aranan Olasılık

İstenen olasılık, testin pozitif çıktığı bilgisi **verildiğinde** kişinin hasta olma olasılığıdır: $\mathbf{P(\text{hasta} | \text{test poz.})}$.

**Bayes Teoremi Formülü:**
$$P(\text{hasta} | \text{test poz.}) = \frac{P(\text{test poz.} | \text{hasta}) \cdot P(\text{hasta})}{P(\text{test poz.})}$$

---

### 1. Parametrelerin Tanımlanması ve Giriş Verileri

| Veri | Tanım | Değer |
| :--- | :--- | :--- |
| **Önsel Olasılık** | $P(\text{hasta})$ (Hastalık yaygınlığı) | $1\% = \mathbf{0.01}$ |
| **Olabilirlik (True Positive)** | $P(\text{test poz.} | \text{hasta})$ (Hasta iken testin pozitif çıkması) | $95\% = \mathbf{0.95}$ |
| **True Negative** | $P(\text{test neg.} | \text{sağlıklı})$ (Sağlıklı iken testin negatif çıkması) | $90\% = \mathbf{0.90}$ |

---

### 2. Eksik Olasılıkların Hesaplanması (Tümleyen Kuralı)

Bayes formülünü tamamlamak için gerekli değerler:

#### A) Sağlıklı Olma Olasılığı ($P(\text{sağlıklı})$)
$$P(\text{sağlıklı}) = 1 - P(\text{hasta}) = 1 - 0.01 = \mathbf{0.99}$$

#### B) Yanlış Pozitif Olasılığı ($P(\text{test poz.} | \text{sağlıklı})$)
Sağlıklı bir kişide testin pozitif çıkma olasılığı (False Positive):
$$P(\text{test poz.} | \text{sağlıklı}) = 1 - P(\text{test neg.} | \text{sağlıklı}) = 1 - 0.90 = \mathbf{0.10}$$

---

### 3. Kanıtın Hesaplanması ($P(\text{test poz.})$)

Kanıt (payda), hem doğru pozitifler hem de yanlış pozitifler dahil olmak üzere **tüm pozitif test sonuçlarının** toplam olasılığıdır (Toplam Olasılık Kuralı):

$$P(\text{test poz.}) = \big(P(\text{test poz.} | \text{hasta}) \cdot P(\text{hasta})\big) + \big(P(\text{test poz.} | \text{sağlıklı}) \cdot P(\text{sağlıklı})\big)$$

$$P(\text{test poz.}) = (0.95 \cdot 0.01) + (0.10 \cdot 0.99)$$$$P(\text{test poz.}) = 0.0095 + 0.0990$$$$\mathbf{P(\text{test poz.})} = \mathbf{0.1085}$$

---

### 4. Sonsal Olasılığın Hesaplanması (Final)

Tüm değerleri Bayes Teoremi formülüne yerleştirelim:

$$P(\text{hasta} | \text{test poz.}) = \frac{0.0095}{0.1085}$$

$$P(\text{hasta} | \text{test poz.}) \approx \mathbf{0.08755}$$

### ✅ Nihai Sonuç ve Yorum

Test sonucu pozitif çıkan bir kişinin **gerçekten hasta olma olasılığı** yaklaşık olarak **%8.76**'dır ($\mathbf{0.0876}$).

Bu düşük sonuç şaşırtıcıdır ve temel olarak şundan kaynaklanır:
> **Hastalık nadirdir.** Popülasyonun sadece %1'i hastadır. Test oldukça doğru olsa da, çok sayıda sağlıklı insanın yanlış pozitif ($0.10 \cdot 0.99 = 0.0990$) sonuç vermesi, doğru pozitif sonuçlardan ($0.95 \cdot 0.01 = 0.0095$) sayıca çok daha fazladır.
> 

---
---

<img width="966" height="582" alt="image" src="https://github.com/user-attachments/assets/a5111561-ea89-40d6-80df-f1ed8bdea4b9" />

# 🎲 Binom Dağılımı (Binomial Distribution) Problemi Çözümü

Bu problem, sabit sayıda denemeden (20 zar atışı), belirli sayıda (7 kez) başarı elde etme olasılığını hesapladığımız klasik bir **Binom Dağılımı** örneğidir.

### 📝 Binom Dağılımı Formülü

$n$ denemeden $k$ kez başarı elde etme olasılığı:

$$P(X=k) = C(n, k) \cdot p^k \cdot (1-p)^{n-k}$$

---

### 1. Problemin Parametreleri

| Parametre | Tanım | Değer |
| :--- | :--- | :--- |
| **$n$** (Deneme Sayısı) | Zar atışı sayısı | $\mathbf{20}$ |
| **$k$** (İstenen Başarı) | "4" gelme sayısı | $\mathbf{7}$ |
| **$p$** (Başarı Olasılığı) | Bir atışta '4' gelme olasılığı | $\mathbf{1/6}$ |
| **$1-p$** (Başarısızlık Olasılığı) | '4' dışında bir sayı gelme olasılığı | $1 - 1/6 = \mathbf{5/6}$ |
| **$n-k$** (Başarısızlık Sayısı) | Başarısız atış sayısı | $20 - 7 = \mathbf{13}$ |

---

### 2. Formülün Uygulanması

Bulunan değerler formülde yerine konur. $C(20, 7)$ ifadesi, kombinasyon gösterimi olan $\binom{20}{7}$ şeklinde yazılır:

$$P(X=7) = \binom{20}{7} \cdot \left(\frac{1}{6}\right)^7 \cdot \left(\frac{5}{6}\right)^{13}$$

### ✅ Doğru Seçenek

Bu matematiksel ifadeye karşılık gelen seçenek, **dördüncü (alttan birinci) seçenektir**.

$$P(X=7) = \binom{20}{7} \cdot \left(\frac{1}{6}\right)^7 \cdot \left(\frac{5}{6}\right)^{13}$$

---
---

<img width="763" height="616" alt="image" src="https://github.com/user-attachments/assets/f69d8cc5-702d-47bc-9460-cc0addb2f6fd" />

# 🚕 Ayrık Olasılık Çözümü: Kümülatif Dağılım

Bu, bir taksi yolculuğunda yolcu sayısının ($X$) **3 veya daha az** olma olasılığını hesaplayan bir **Ayrık Olasılık Dağılımı** problemidir. İstenen olasılık $\mathbf{P(X \le 3)}$'tür.

### 📝 Çözüm: Kümülatif Olasılık Hesaplama

$P(X \le 3)$ olasılığı, $X$'in 0, 1, 2 veya 3 olduğu ayrı olasılıkların toplamına eşittir:

$$P(X \le 3) = P(X=0) + P(X=1) + P(X=2) + P(X=3)$$

### 1. Tablodan Olasılıkları Alma

| Yolcu Sayısı ($x_i$) | Olasılık ($p_i$) |
| :--- | :--- |
| $X=0$ | $0.10$ |
| $X=1$ | $0.25$ |
| $X=2$ | $0.25$ |
| $X=3$ | $0.15$ |

### 2. Olasılıkları Toplama

İstenen kümülatif olasılığı bulmak için değerleri toplarız:

$$P(X \le 3) = 0.10 + 0.25 + 0.25 + 0.15$$

$$\mathbf{P(X \le 3)} = \mathbf{0.75}$$

### ✅ Sonuç

Rastgele seçilen bir taksi yolculuğunda yolcu sayısının 3 veya daha az olma olasılığı $\mathbf{0.75}$'tir (veya **%75**).

---
---

<img width="827" height="529" alt="image" src="https://github.com/user-attachments/assets/c48b6a1e-6605-4e0e-a500-d76c386a4d64" />

<img width="674" height="524" alt="image" src="https://github.com/user-attachments/assets/4c938733-bd84-49ce-b1d1-f1ebfef7b961" />

# 📉 Dört Normal Dağılımın (Gaussian) Karşılaştırmalı Analizi

Bu analiz, dört farklı Normal Dağılımın (Çan Eğrisi) iki temel parametresi olan **Ortalama ($\mu$)** ve **Standart Sapma ($\sigma$)** değerlerinin grafikte nasıl yorumlandığını gösterir.

### 🧠 Normal Dağılım Parametrelerinin Yorumlanması

* **Ortalama ($\mu$) Konumu:** Eğrinin **en yüksek noktası (tepesi)** $\mu$ değerine denk gelir.
* **Standart Sapma ($\sigma$) Yayılımı:** Eğri ne kadar **dar ve uzunsa**, $\sigma$ o kadar **küçüktür** (düşük varyans). Eğri ne kadar **geniş ve alçaksa**, $\sigma$ o kadar **büyüktür** (yüksek varyans).

---

### 1. Dağılımların Detaylı Analizi Tablosu

| Dağılım | Renk/Çizgi Tipi | Ortalama ($\mu$) Yorumu (Tepe Noktası) | Standart Sapma ($\sigma$) Yorumu (Genişlik/Yükseklik) |
| :--- | :--- | :--- | :--- |
| **normal\_A** | Mavi, Düz | $\mathbf{x \approx -2.5}$ civarındadır. | En **dar** ve en **yüksek** eğridir. **En küçük $\sigma$** değerine sahiptir. |
| **normal\_B** | Turuncu, Kesik | $\mathbf{x \approx 0}$ civarındadır. | normal\_A'dan daha geniş, ancak diğerlerinden küçüktür. |
| **normal\_C** | Yeşil, Nokta-Kesik | $\mathbf{x \approx 1.5}$ civarındadır. | normal\_B'den daha geniştir. |
| **normal\_D** | Kırmızı, Noktalı | $\mathbf{x \approx 4}$ civarındadır. | En **geniş** ve en **alçak** eğridir. **En büyük $\sigma$** değerine sahiptir. |

---

### 2. Sıralamaların Özeti

| Parametre | Sıralama (Küçükten Büyüğe) |
| :--- | :--- |
| **Ortalama ($\mu$)** | $\mu_{\text{normal\_A}} < \mu_{\text{normal\_B}} < \mu_{\text{normal\_C}} < \mu_{\text{normal\_D}}$ |
| **Standart Sapma ($\sigma$)** | $\sigma_{\text{normal\_A}} < \sigma_{\text{normal\_B}} < \sigma_{\text{normal\_C}} < \sigma_{\text{normal\_D}}$ |

---

### 3. ✅ Doğru İfadelerin Seçimi

Bu sıralamalara göre, grafiği doğru yorumlayan ifadeler şunlardır:

* $$\mathbf{\sigma_{\text{normal\_D}} > \sigma_{\text{normal\_A}}}$$
    (En geniş olan D, en dar olan A'dan büyüktür.)

* $$\mathbf{\mu_{\text{normal\_D}} > \mu_{\text{normal\_C}}}$$
    (En sağdaki D, C'den daha büyük ortalamaya sahiptir.)

* $$\mathbf{\sigma_{\text{normal\_C}} > \sigma_{\text{normal\_B}}}$$
    (C, B'den daha yayvan/geniş bir dağılıma sahiptir.)


---
<img width="1199" height="587" alt="image" src="https://github.com/user-attachments/assets/f66e93f1-f141-456d-8302-054b7308bc2f" />

### Fayda: Dağılımın Standartlaştırılmasının Faydaları

| Benefit (Concept) | Explanation / Impact |
| :--- | :--- |
| **Standard Scale Transformation** 📏 | It transforms datasets into a standard scale, making it easier to compare between different datasets. |
| **Statistical Simplification** 📊 | It simplifies statistical analysis, particularly when using techniques that assume a standard normal distribution. |
| **Machine Learning Performance** 🚀 | Standardizing features in machine learning can improve the convergence rate of optimization algorithms and prevent some features from dominating others, leading to improved model performance. |
---

<img width="717" height="243" alt="image" src="https://github.com/user-attachments/assets/df2a2ea2-7d39-49a5-a0e9-afaebd94f442" />

### Kurtosis Analizi: Game A vs. Game B Dağılım Karşılaştırması ⚖️

Kurtosis, bir dağılımın **kuyruklarının ağırlığını** ve aşırı uç değerlere sahip olma eğilimini ölçer. Yüksek Kurtosis, daha ağır kuyruklar demektir.

| Kriter (Criterion) | Game A Analizi (Game A Analysis) | Game B Analizi (Game B Analysis) |
| :--- | :--- | :--- |
| **Kazanma/Kaybetme Değerleri** | $\{-1, +2\}$ | $\{-2, -0.50, +0.50, +5\}$ |
| **Aralık (Range)** | Küçük aralık ($\text{Range} = 3$ birim). | Büyük aralık ($\text{Range} = 7$ birim). |
| **Uç Değerler (Extremes)** | Tüm olası sonuçlar birbirine nispeten yakındır. | Çok daha **uç sonuçlar** ($\text{+5}$ ve $\text{-2}$) mevcuttur. |
| **Kuyruk Ağırlığı (Tail Weight)** | Kuyruklar **hafif** olma eğilimindedir. | Kuyruklar **ağır** olma eğilimindedir (daha fazla aşırı değer olasılığı). |
| **Kurtosis Sonucu** | Daha küçüktür ($\text{Smaller}$). | Daha büyüktür ($\text{Larger}$). |

#### Nihai Sonuç (Final Conclusion)

<img width="742" height="406" alt="image" src="https://github.com/user-attachments/assets/f7b0f71d-02a3-45e1-8d98-b306052c9c88" />


Game B, Game A'ya kıyasla dağılımının merkezinden çok daha uzakta yer alan büyük uç değerlere sahip olduğu için, **Game B'nin kurtosis değeri, Game A'nın kurtosis değerinden daha büyüktür.**

$$\text{Game A's kurtosis is smaller than Game B's kurtosis.}$$
$$\text{Kurtosis}(\text{A}) < \text{Kurtosis}(\text{B})$$

---

<img width="521" height="455" alt="image" src="https://github.com/user-attachments/assets/979a8b6f-a4af-4fcb-b5ef-b1f61ba89819" />

### Bağımsız Normal Dağılımlı Değişkenlerin Toplamı Analizi ➕

**Verilenler (Given):**
* $X \sim \text{Normal}(3, 1^2)$
* $Y \sim \text{Normal}(2, 2^2)$

**Kural (Rule):** $X$ ve $Y$ bağımsız ise, $Z = X + Y$ de Normal dağılım izler: $Z \sim \text{Normal}(\mu_Z, \sigma_Z^2)$.

---

#### 1. Ortalama ($\mu_Z$) Hesaplanması (Mean Calculation)

Bağımsız değişkenlerin ortalaması, bireysel ortalamaların toplamıdır:
$$\mu_Z = \mu_X + \mu_Y$$

| Değişken | Ortalama ($\mu$) |
| :---: | :---: |
| $X$ | $\mu_X = 3$ |
| $Y$ | $\mu_Y = 2$ |

$$\mu_Z = 3 + 2 = 5$$

---

#### 2. Varyans ($\sigma_Z^2$) Hesaplanması (Variance Calculation)

Bağımsız değişkenlerin varyansı, bireysel varyansların toplamıdır:
$$\sigma_Z^2 = \sigma_X^2 + \sigma_Y^2$$

| Değişken | Varyans ($\sigma^2$) |
| :---: | :---: |
| $X$ | $\sigma_X^2 = 1^2 = 1$ |
| $Y$ | $\sigma_Y^2 = 2^2 = 4$ |

$$\sigma_Z^2 = 1 + 4 = 5$$

---

#### 3. Standart Sapma ($\sigma_Z$) Hesaplanması (Standard Deviation)

Standart sapma, varyansın kareköküdür:
$$\sigma_Z = \sqrt{\sigma_Z^2}$$

$$\sigma_Z = \sqrt{5}$$

---

#### Sonuç (Final Result)

$Z = X + Y$ değişkeni, $\text{Normal}(\mu, \sigma^2)$ dağılımına sahiptir:
$$\mu = 5, \quad \sigma = \sqrt{5}$$

---

<img width="680" height="646" alt="image" src="https://github.com/user-attachments/assets/8995caa0-e725-45a3-a22e-c8194d872070" />

### Kutu Grafiği Analizi: Class A vs. Class B Test Skorları 📊

Bu analizde, medyan (merkez) ve çeyrekler arası aralık (yayılım) karşılaştırılmıştır.

| İstatistiksel Ölçü | Tanım (Definition) | Class A Değerleri | Class B Değerleri |
| :--- | :--- | :--- | :--- |
| **Medyan (Median)** 🔴 | Kutunun içindeki yatay kırmızı çizgidir. | Yaklaşık $\mathbf{74}$ | Yaklaşık $\mathbf{85}$ |
| **IQR (Çeyrekler Arası Aralık)** 📐 | Kutunun yüksekliğidir ($\text{Q3} - \text{Q1}$). | $\text{Q3} \approx 79, \text{Q1} \approx 60 \implies \text{IQR} \approx \mathbf{19}$ | $\text{Q3} \approx 90, \text{Q1} \approx 80 \implies \text{IQR} \approx \mathbf{10}$ |

---

#### Temel Sonuçlar (Key Findings)

* **Medyan Karşılaştırması:** Class B'nin medyan skoru ($\mathbf{85}$), Class A'nın medyan skorundan ($\mathbf{74}$) **daha yüksektir**. ($\text{Class B}$ daha iyi bir merkezi eğilime sahiptir.)
* **IQR Karşılaştırması:** Class A'nın IQR'ı ($\mathbf{19}$), Class B'nin IQR'ından ($\mathbf{10}$) **daha büyüktür**. ($\text{Class A}$'nın skorları daha fazla yayılmıştır/dağınıktır.)

### Kutu Grafiği Analizi: İfadelerin Karşılaştırılması (Comparison of Statements) 🎯

Aşağıdaki tablo, Class A ve Class B için hesaplanan Medyan ve IQR değerlerine göre verilen ifadelerin doğruluğunu kontrol etmektedir.

| İfade (Statement) | Değerler (Values) | Sonuç (Result) |
| :--- | :--- | :--- |
| **Class A's median score is higher than Class B's median score.** | $74 > 85$ | Yanlış (False) ❌ |
| **Class B's interquartile range (IQR) is larger than Class A's interquartile range.** | $10 > 19$ | Yanlış (False) ❌ |
| **Class A's interquartile range (IQR) is larger than Class B's interquartile range.** | $19 > 10$ | **Doğru (True)** ✅ |
| **Class B's median score is higher than Class A's median score.** | $85 > 74$ | **Doğru (True)** ✅ |

#### Analiz Özeti

Grafiğe göre, **Class B** daha yüksek bir merkezi eğilime (Medyan = 85) sahipken, **Class A** daha büyük bir yayılıma ($\text{IQR} = 19$) sahiptir.

### Kutu Grafiği Sonuçları: Doğru İfadeler (Final Box Plot Conclusions) ✅

Analiz sonucunda, verilen dört seçenek arasından **iki ifadenin** doğru olduğu belirlenmiştir. Bu durum, veri setinin hem merkezindeki hem de yayılımındaki farklılıkları yansıtmaktadır.

| İfade (Statement) | Dayanak (Evidence) | Sonuç (Result) |
| :--- | :--- | :--- |
| **Class A's interquartile range (IQR) is larger than Class B's interquartile range.** | $\text{IQR(A): 19} \quad > \quad \text{IQR(B): 10}$ | **Doğru (True)** |
| **Class B's median score is higher than Class A's median score.** | $\text{Median(B): 85} \quad > \quad \text{Median(A): 74}$ | **Doğru (True)** |

---

#### Özet (Summary)

Class B'nin **daha yüksek bir performansa** (daha yüksek medyan) sahip olduğu, ancak Class A'nın skorlarının **daha dağınık** (daha büyük IQR) olduğu sonucuna varılmıştır.

---

<img width="560" height="599" alt="image" src="https://github.com/user-attachments/assets/d7d014ed-90f3-4423-b5c5-f7d020806d74" />

### QQ Plot Analizi: Normal Dağılımın Değerlendirilmesi 📐📊

Bu analiz, bir veri setinin **QQ Plot** (Kuantil-Kuantil Grafiği) kullanılarak Normal ($\text{Gaussian}$) dağılıma ne kadar uyduğunu değerlendirmektedir.

| Kriter (Criterion) | Gözlem (Observation) | Sonuç (Implication) |
| :--- | :--- | :--- |
| **Genel Kural** | Veri noktaları düz çizgiyi takip ediyorsa, veri Normal dağılıma uyar. Veri noktaları çizgiden sapıyorsa, dağılım göstermez. | QQ Plot, verinin dağılım uyumunu belirlemenin en iyi yoludur. |
| **Merkez Bölge** 🟢 | Veri noktalarının büyük bir kısmı ($-1$'den $1$'e kadar olan kısım), turuncu çizginin üzerinde çok yakındır. | Verinin merkezinin Normal dağılıma **güçlü bir şekilde** uyduğunu gösterir. |
| **Kuyruklar (Uçlar)** ⚠️ | Noktalar, hem alt kuyrukta (yaklaşık $-1$'den sonra) hem de üst kuyrukta (yaklaşık $1$'den sonra) çizgiden hafifçe sapmaktadır. | Dağılımın kusursuz Normal olmadığını gösterir, ancak sapma küçüktür. |
| **Nihai Karar** ✅ | Sapmaların büyüklüğü, tüm veri setini "Normal Dağılmış Değil" olarak sınıflandırmak için genellikle yeterli değildir. | **The data looks normally distributed.** (Veri, Normal dağılmış görünüyor.) |

---

<img width="1077" height="490" alt="image" src="https://github.com/user-attachments/assets/2d51ea35-a613-4ae1-9d9c-8db533ba4c7b" />

---

<img width="1076" height="492" alt="image" src="https://github.com/user-attachments/assets/4b7028bb-a6b7-4750-b775-4937b2213b2e" />

---

<img width="361" height="149" alt="image" src="https://github.com/user-attachments/assets/dc036e40-7f62-4faf-88c7-6b21cf5e2af5" />

---
<img width="706" height="335" alt="image" src="https://github.com/user-attachments/assets/0b875d7d-3e2a-46c0-a74f-107855a6e3a5" />

### Ayrık Olasılık Dağılımının Beklenen Değeri ($E[X]$) Hesaplaması 🧮

Bu problem, rastgele değişken $X$'in alacağı değerlerin kendi olasılıkları ile çarpılıp toplanmasıyla bulunan **Beklenen Ortalama** ($\mu$) değerini hesaplamayı gerektirir.

#### Temel Formül (Expected Value Formula)
$$\mu = E[X] = \sum_{i} x_i \cdot P(x_i)$$

---

#### 1. Veri Değerleri ve Olasılıklar Tablosu 🔢

| $X$ Değeri ($x_i$) | Olasılık ($P(x_i)$) |
| :---: | :---: |
| $1$ | $0.3$ |
| $3$ | $0.4$ |
| $5$ | $0.3$ |

---

#### 2. Beklenen Değer ($E[X]$) Hesaplaması

Her $x_i$ değerini karşılık gelen $P(x_i)$ olasılığı ile çarpıp toplayalım:

$$E[X] = (1 \cdot 0.3) + (3 \cdot 0.4) + (5 \cdot 0.3)$$

$$E[X] = 0.3 + 1.2 + 1.5$$

$$E[X] = 3.0$$

---

#### Sonuç (Final Result)

Beklenen ortalama ($\mu$) şuna eşittir:
$$\mu = 3.0$$

---

<img width="709" height="414" alt="image" src="https://github.com/user-attachments/assets/3de533ee-51e0-43d5-89a5-2750bbc2fc8a" />

### Ortak Olasılık Dağılımı Analizi: Çift Değer Olasılığı ($P(X=\text{even}, Y=\text{even})$) 🎲

Bu problem, iki rastgele değişkenin ($X$ ve $Y$) **bir arada çift değer** alması olasılığını bulmayı gerektirir.

---

#### 1. Koşulun Belirlenmesi (Defining the Condition)

| Değişken (Variable) | Olası Değerler (Possible Values) | Çift Değer (Even Value) |
| :---: | :---: | :---: |
| $X$ | $\{1, 2, 3\}$ | $X=2$ |
| $Y$ | $\{1, 2\}$ | $Y=2$ |

Koşul: $X$ ve $Y$ her ikisi de çift değer alsın.
$$\text{Aranan Olasılık: } P(X=2, Y=2)$$

---

#### 2. Olasılığın Tablodan Okunması (Reading the Joint Probability)

Verilen ortak olasılık dağılımı tablosu:

| $X \setminus Y$ | $1$ | $2$ | $3$ |
| :---: | :---: | :---: | :---: |
| $1$ | $0.1$ | $0.2$ | $0.3$ |
| $2$ | $0.2$ | **0.1** | $0.1$ |

Tabloda $X=2$ satırı ile $Y=2$ sütununun kesişimi aranan olasılıktır:

$$P(X=2, Y=2) = 0.1$$

---

#### Nihai Sonuç (Final Result) ✅

Doğru hesaplama ve tablodan okunan değer şudur:

$$\mathbf{P(X=2, Y=2) = 0.1}$$

---

<img width="699" height="354" alt="image" src="https://github.com/user-attachments/assets/43b4b95f-fd08-4fdd-aa5f-fccf8190a6ea" />


### Koşullu Olasılık Hesaplaması: $P(X=3 \mid Y=1)$ 🎯

Bu problem, ortak olasılık dağılımı tablosu kullanılarak koşullu olasılığın hesaplanmasını gerektirir.

#### Koşullu Olasılık Formülü
$$P(X=x \mid Y=y) = \frac{P(X=x, Y=y)}{P(Y=y)}$$

---

#### 1. Payda Hesaplaması: Marjinal Olasılık $P(Y=1)$ ➕

$P(Y=1)$, $Y=1$ sütunundaki tüm değerlerin toplamıdır:

| $X$ Değeri | $P(X, Y=1)$ |
| :---: | :---: |
| 1 | $0.05$ |
| 2 | $0.10$ |
| 3 | $0.15$ |
| **Toplam** | $\mathbf{0.30}$ |

$$P(Y=1) = 0.05 + 0.10 + 0.15 = 0.30$$

---

#### 2. Pay Değeri: Ortak Olasılık $P(X=3, Y=1)$

Tablodan okunan değer:
$$P(X=3, Y=1) = 0.15$$

---

#### 3. Nihai Hesaplama (Final Calculation)

$$P(X=3 \mid Y=1) = \frac{P(X=3, Y=1)}{P(Y=1)} = \frac{0.15}{0.30}$$

$$\mathbf{P(X=3 \mid Y=1) = 0.5}$$

---

<img width="731" height="167" alt="image" src="https://github.com/user-attachments/assets/52ab6daf-7543-4fc5-937b-cb656964b808" />


### Kovaryans Hesaplaması: $\text{Cov}(X, Y)$ 🔗

Bu analiz, $\text{Cov}(X, Y) = E[XY] - E[X]E[Y]$ formülüne dayanmaktadır.

#### Ortak Olasılık Dağılımı Tablosu
| $X \setminus Y$ | 0 | 1 | **$P(X=x)$** |
| :---: | :---: | :---: | :---: |
| 0 | $0.2$ | $0.1$ | $0.3$ |
| 1 | $0.1$ | $0.6$ | $0.7$ |
| **$P(Y=y)$** | $0.3$ | $0.7$ | **1.0** |

---

#### 1. Beklenen Değerlerin Hesaplanması (Expected Values)

* **$E[X]$:** $(0 \cdot 0.3) + (1 \cdot 0.7) = \mathbf{0.7}$
* **$E[Y]$:** $(0 \cdot 0.3) + (1 \cdot 0.7) = \mathbf{0.7}$

---

#### 2. Çarpımın Beklenen Değeri ($E[XY]$)

| $x$ | $y$ | $x \cdot y$ | $P(x, y)$ | $(x \cdot y) \cdot P(x, y)$ |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 0 | 0 | 0.2 | 0.0 |
| 0 | 1 | 0 | 0.1 | 0.0 |
| 1 | 0 | 0 | 0.1 | 0.0 |
| 1 | 1 | 1 | 0.6 | 0.6 |

$$E[XY] = 0.0 + 0.0 + 0.0 + 0.6 = \mathbf{0.6}$$

---

#### 3. Kovaryansın Hesaplanması (Final Covariance)

$$\text{Cov}(X, Y) = E[XY] - E[X] E[Y]$$
$$\text{Cov}(X, Y) = 0.6 - (0.7) \cdot (0.7)$$
$$\text{Cov}(X, Y) = 0.6 - 0.49$$

$$\mathbf{\text{Cov}(X, Y) = 0.11}$$

---

# 📈 Merkezi Limit Teoremi (CLT) vs. Büyük Sayılar Yasası (LLN) Karşılaştırması

| Özellik | Merkezi Limit Teoremi (CLT) 💡 | Büyük Sayılar Yasası (LLN) ⚖️ |
| :--- | :--- | :--- |
| **Temel Tanım** | Bir popülasyondan alınan bağımsız ve aynı dağılıma sahip **örneklem ortalamalarının dağılımı**, örneklem büyüklüğü ($n$) yeterince arttıkça, popülasyonun orijinal dağılımı ne olursa olsun, **Normal Dağılıma** yaklaşır. | Örneklem büyüklüğü ($n$) arttıkça, **örneklem ortalaması ($\bar{x}$)**, popülasyonun gerçek **ortalamasına ($\mu$)** yaklaşır ve ona yakın kalır. |
| **Odak Noktası** | **Dağılımın Şekli** (Örneklem ortalamalarının dağılımı nasıl görünür?) | **Ortalamanın Değeri** (Örneklem ortalaması nereye gider?) |
| **İstenen $n$** | Dağılımın Normal'e yakınsaması için genellikle $n \ge 30$ olması beklenir. | Ne kadar büyük olursa o kadar iyi; yakınsama için tek şart $n \to \infty$ (sonsuz) olmasıdır. |
| **Matematiksel İfade** | $$\bar{X} \xrightarrow{d} N(\mu, \sigma^2/n)$$ (Dağılım, Normal Dağılıma yakınsar) | $$\bar{X} \xrightarrow{p} \mu$$ (Ortalama, değere olasılıkla yakınsar) |
| **Varyans İlişkisi** | Örnekleme dağılımının varyansı $\sigma^2/n$'dir, bu da $n$ arttıkça **dağılımın daraldığı** anlamına gelir (Daha kesin tahmin). | LLN, esas olarak merkezi eğilimle ilgilenir, varyansın azalması CLT'nin bir sonucudur. |
| **Pratik Anlamı** | İstatistiksel testlerin (t-testi, Z-testi) ve güven aralıklarının oluşturulmasının temelini sağlar. Popülasyon bilinmese bile çıkarım yapmayı mümkün kılar. | Uzun vadede gözlemlenen sonucun teorik beklentiye eşit olacağını garanti eder (Ör. 1000 kez yazı tura atıldığında yazı gelme oranının %50'ye yaklaşması). |
| **Örnek** | Bir madeni parayı 30 kez çevirip ortalama tura gelme sayısını defalarca kaydettiğinizde, bu ortalamaların bir histogramı çan eğrisine benzer. | Bir madeni parayı defalarca çevirdiğinizde, tura gelme oranının $\frac{1}{2}$ değerine giderek daha çok yaklaşması. |

### Metaforik Karşılaştırma (Örnekler) 🎲🎯

| Teorem | Metafor | Açıklama |
| :--- | :--- | :--- |
| **Merkezi Limit Teoremi (CLT) 📏** | **Oyunun Kuralları** | Bir zar oyunu oynarken, tek bir zarın sonucu rastgele (düzgün dağılım). Ancak **çok sayıda** tur oynarsanız ve her turdaki ortalama puanı toplarsanız, bu ortalama puanların grafiği bir **çan eğrisi** (normal dağılım) şeklini alır. CLT bize, oyunun kuralları (dağılımın şekli) ne olursa olsun, *büyük serilerin* hep aynı şekilde davrandığını söyler. |
| **Büyük Sayılar Yasası (LLN) 🎯** | **Hedefe Ulaşma** | Bir hedef tahtasına atış yapıyorsunuz. İlk birkaç atışınız rastgele yerlere düşebilir. Ancak atış sayınızı **binlerceye** çıkardığınızda, atışlarınızın **ortalaması** (merkezi) hedefin tam ortasına (gerçek popülasyon ortalamasına) giderek daha çok yaklaşacaktır. LLN, bize yeterince deneme yaparsak **hedefi vuracağımızı** garanti eder. |

---

# Maksimum Olabilirlik Tahmincisi (Maximum Likelihood Estimation - MLE)
## Gauss Popülasyonu İçin MLE (MLE for Gaussian Population)


### Matematiksel Formülasyon (Mathematical Formulation)
Ortalaması $\mu$ ve varyansı $\sigma^2$ olan bir Gauss dağılımından alınan $n$ örnekleme $X=(X_1, X_2, \dots, X_n)$ sahip olduğunuzu varsayalım. Bu, $X_i \sim_{i.i.d.} N(\mu, \sigma^2)$ anlamına gelir.

Eğer $\mu$ ve $\sigma$ için MLE istiyorsanız, ilk adım **Olabilirlik Fonksiyonu (Likelihood)**'nu tanımlamaktır. Eğer hem $\mu$ hem de $\sigma$ bilinmiyorsa, olabilirlik bu iki parametrenin bir fonksiyonu olacaktır. $x=(x_1, x_2, \dots, x_n)$ ile verilen $X$'in bir gerçekleşimi (realization) için:

$$ L(\mu,\sigma; \boldsymbol{x}) = \prod_{i=1}^n f_{X_i}(x_i) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma } e^{-\frac{1}{2}\frac{(x_i-\mu)^2}{\sigma^2}} $$
$$ L(\mu,\sigma; \boldsymbol{x}) = \frac{1}{(\sqrt{2\pi})^n\sigma^n }e^{-\frac{1}{2}\frac{\sum_{i=1}^n (x_i-\mu)^2}{\sigma^2}} $$

Şimdi yapmanız gereken tek şey, olabilirlik $L(\mu, \sigma; \boldsymbol{x})$'i maksimize eden $\mu$ ve $\sigma$ değerlerini bulmaktır.

### Log-Olabilirlik (Log-Likelihood) Fonksiyonu

Olabilirlik fonksiyonunun türevini almak karmaşık olduğu için, logaritma fonksiyonunun her zaman artan olmasından faydalanarak **Log-Olabilirlik Fonksiyonu** kullanılır:

$$ \ell(\mu,\sigma) = \log(L(\mu,\sigma; \boldsymbol{x})) $$

Logaritmanın çarpımı toplama dönüştürme özelliğini ($\log(a \cdot b) = \log(a) + \log(b)$) ve diğer logaritma özelliklerini kullanarak Log-Olabilirlik şu şekilde basitleştirilir:

$$ \ell(\mu,\sigma) = -\frac{n}{2}\log(2\pi) - n\log(\sigma) - \frac{1}{2}\frac{\sum_{i=1}^n (x_i-\mu)^2}{\sigma^2} $$

### MLE'nin Türetilmesi (Derivation of MLE)

MLE için $\mu$ ve $\sigma$ değerlerini bulmak için, Log-Olabilirlik'in kısmi türevleri (partial derivatives) alınır ve sıfıra eşitlenir.

#### a) $\mu$ İçin Kısmi Türev ($\partial / \partial \mu$):

$$\frac{\partial }{\partial \mu}\ell(\mu, \sigma) = \frac{1}{\sigma^2}\left(\sum_{i=1}^n x_i - n\mu\right) = 0$$

$\sigma > 0$ olduğu için $\sum_{i=1}^n x_i - n\mu = 0$ olmalıdır. Buradan $\mu$ için MLE tahmini:

$$\hat{\mu} = \frac{\sum_{i=1}^n x_i}{n} = \bar{x}$$
**Sonuç:** Ortalama için MLE, **Örneklem Ortalaması (Sample Mean)**'dır.

#### b) $\sigma$ İçin Kısmi Türev ($\partial / \partial \sigma$):

$$\frac{\partial }{\partial \sigma}\ell(\mu, \sigma) = -\frac{n}{\sigma} + \left(\sum_{i=1}^n (x_i-\mu)^2\right)\frac{1}{\sigma^3} = 0$$

$\mu$'yu $\hat{\mu}=\bar{x}$ ile değiştirip $\sigma > 0$ olduğu için ifadeyi basitleştirirsek:

$$\frac{\partial }{\partial \sigma}\ell(\mu, \sigma) = -n + \left(\sum_{i=1}^n (x_i-\bar{x})^2\right)\frac{1}{\sigma^2} = 0$$

Buradan varyans için MLE tahmini:

$$\hat{\sigma}^2 = \frac{\sum(x_i-\bar{x})^2}{n}$$

**Sonuç:** Standart sapma için MLE ($\hat{\sigma}$), bu ifadenin kareköküdür. Bu ifade, örneklem standart sapması (sample standard deviation) için öğrendiğiniz formüle çok benzerdir, tek fark **$1/n$** ile normalleştirme yapılmasıdır. Örneklem standart sapması ise **$1/(n-1)$** kullanır.
---

# 📊 Veri Bilimi ve Makine Öğrenimi Öğrenim Planı

Bu belge, Veri Bilimi ve Makine Öğrenimi alanındaki Coursera eğitimlerini mantıksal aşamalara göre gruplandırılmış bir çalışma planını sunar.

---

## 🚀 Maksimum Olabilirlik Tahmincisi (MLE) Açıklaması

MLE, eldeki veriyi en olası (highest likelihood) kılan model parametrelerini bulma yöntemidir. ML'de Çapraz Entropi gibi maliyet fonksiyonlarının temelini oluşturur.

### MLE vs. En Küçük Kareler (Least Squares) Karşılaştırması

| Kriter | Maksimum Olabilirlik Tahmincisi (MLE) 🎯 | En Küçük Kareler (Least Squares - LS) 📏 |
| :--- | :--- | :--- |
| **Temel Felsefe** | Olasılığa (Likelihood) dayanır. | Hataların karesini minimize etmeye dayanır. |
| **Hesaplama Amacı** | Gözlemleri **en olası** yapan $\theta$ parametrelerini bulmak. | Tahminler ($ \hat{y} $) ile gerçek değerler ($ y $) arasındaki **mesafeyi** minimize etmek. |
| **Gereken Varsayım** | Verinin **olasılık dağılımını** bilmek zorunludur (Normal, Bernoulli vb.). | Hatanın dağılımı hakkında açık bir varsayım yapmaz. |
| **ML'deki Karşılığı** | Çapraz Entropi Kaybı (Cross-Entropy Loss), Lojistik Regresyon. | Ortalama Karesel Hata (Mean Squared Error - MSE). |
| **Eşitlik Durumu** | Hata terimleri **Normal Dağılım'a** sahipse, MLE'yi maksimize etmek, LS'yi minimize etmeye eşdeğerdir. |

---

# 🤖 Makine Öğreniminde Maksimum Olabilirlik (Maximum Likelihood Estimation - MLE)

### Amaç: 🎯

MLE'nin temel amacı, elimizdeki gözlemlenen veriyi ($\mathbf{X}$) en olası (en yüksek olasılıklı) hale getiren model parametrelerini ($\theta$) bulmaktır.

Başka bir deyişle, "Eğer modelimin parametreleri $\theta$ olsaydı, bu veriyi görme olasılığım ne olurdu?" sorusuna cevap vererek bu olasılığı maksimize etmektir.


---

### ML'de Kullanım Alanı ve Rolü: 🛠️

MLE, genellikle modelin çıktısının olasılıksal (probabilistic) olarak modellendiği durumlarda bir **Maliyet Fonksiyonu (Loss Function)** olarak kullanılır.

#### 1. Sınıflandırma (Classification):

* Özellikle **Lojistik Regresyon (Logistic Regression)** ve **Yapay Sinir Ağları (Neural Networks)** gibi modellerde, tahmin edilen çıktının bir olasılık dağılımı (örneğin, Bernoulli veya Kategorik dağılım) olduğu varsayılır.
* Bu modellerde kullanılan **Çapraz Entropi Kaybı (Cross-Entropy Loss)**, aslında MLE'nin bir uygulamasıdır. Çapraz Entropi'yi minimize etmek, doğru sınıfı gözlemleme olasılığını maksimize eden parametreleri bulmaya eşdeğerdir.

#### 2. Regresyon (Regression):

* Eğer modelin hatalarının (rezidüellerinin) **Normal Dağılım'a (Gaussian Distribution)** sahip olduğu varsayılırsa, **En Küçük Kareler (Least Squares)** yönteminin uygulanması, matematiksel olarak MLE'nin uygulanmasıyla aynı parametre tahminlerini verir (yukarıdaki türetme örneğinde olduğu gibi).

#### 3. Üretici Modeller (Generative Models):

* Veri kümesinin tamamının nasıl oluşturulduğunu öğrenen (örneğin, **Naif Bayes (Naïve Bayes)**) modeller, parametrelerini tahmin etmek için sıklıkla MLE'yi kullanır.

### ⚖️ MLE ve En Küçük Kareler (Least Squares - LS) Karşılaştırması

En Küçük Kareler (LS), genellikle regresyonda kullanılan basit bir maliyet fonksiyonudur.

| Kriter | Maksimum Olabilirlik Tahmincisi (MLE) 🎯 | En Küçük Kareler (Least Squares - LS) 📏 |
| :--- | :--- | :--- |
| **Temel Felsefe** | Olasılığa (Likelihood) dayanır. | Hataların karesini minimize etmeye dayanır. |
| **Hesaplama Amacı** | Gözlemleri **en olası** yapan $\theta$ parametrelerini bulmak. | Tahminler ($\hat{y}$) ile gerçek değerler ($y$) arasındaki **mesafeyi** minimize etmek. |
| **Gereken Varsayım** | Verinin (veya hatanın) **olasılık dağılımını** bilmek zorunludur (örneğin: Normal, Bernoulli, Poisson). | Hatanın dağılımı hakkında **açık bir varsayım yapmaz**, sadece varyansın sabit olduğunu varsayar (Homoscedasticity). |
| **ML'deki Karşılığı** | Çapraz Entropi (Cross-Entropy Loss), Lojistik Regresyon, Yapay Sinir Ağları (NN). | Ortalama Karesel Hata (Mean Squared Error - MSE) ve Basit Lineer Regresyon. |
| **Eşitlik Durumu** | Eğer hata terimleri **Normal Dağılım'a** sahipse, MLE'yi maksimize etmek, LS'yi minimize etmeye eşdeğerdir. |


# 🔬 Sıklıkçı Yaklaşım (Frequentist Statistics)

Bu yaklaşım, olasılık ve bilgi kavramlarını tamamen farklı yorumlayan iki ana ekolden (Sıklıkçı ve Bayesçi) biridir.

Sıklıkçılık, olasılığı, uzun vadede bir olayın gerçekleşme sıklığı olarak görür.

## 1. Sıklıkçı Yaklaşım (Frequentist Statistics) 📏

| Kavram | Açıklama |
| :--- | :--- |
| **Olasılık Anlayışı** | **Uzun Vadeli Olay Sıklığı (Long-term Frequency):** Olasılık, bir deneyi sonsuz kez tekrarladığımızda bir sonucun ne sıklıkla ortaya çıkacağının limitidir. Örnek: Bir madeni paranın tura gelme olasılığı %50'dir, çünkü parayı binlerce kez attığımızda tura gelme sıklığı bu değere yakınsar. |
| **Temel Kavram** | **Olabilirlik (Likelihood):** Elimizdeki veriler (gözlemler) verildiğinde, belirli bir model parametresinin ($\theta$) ne kadar olası olduğunu ölçer. Bu, genellikle $P(\text{Veri} \mid \theta)$ olarak ifade edilir. Sıklıkçılar, sadece veriye bakarak çalışır. |
| **Amaç** | **Veriyi En Olası Üreten Modeli Bulmak:** Amaç, gözlemlenen veriyi en iyi açıklayan ve en yüksek olabilirlik değerini veren **sabit model parametrelerini** bulmaktır (Örn: p-değerleri, Güven Aralıkları hesaplama). |
| **Parametreler** | Popülasyon parametreleri ($\mu, \sigma$ vb.) **sabit ancak bilinmeyen** değerler olarak kabul edilir. |

# 🧠 Bayesçi Yaklaşım (Bayesian Statistics)

Bayesçilik, olasılığı, bilinmeyene olan kişisel inancın veya kesinliğin derecesi olarak görür.

## 2. Bayesçi Yaklaşım (Bayesian Statistics) 💡

| Kavram | Açıklama |
| :--- | :--- |
| **Olasılık Anlayışı** | **İnanç Derecesi (Degree of Belief or Certainty):** Olasılık, bir kişinin veya sistemin, eldeki bilgi ışığında bir önermenin doğru olduğuna ne kadar inandığının sübjektif ölçüsüdür. Yeni bilgi geldikçe bu inanç güncellenir. |
| **Temel Kavram** | **Önsel (Prior) 🤔:** Veri gözlenmeden önce parametrelerin ($\theta$) olası değerleri hakkındaki inancımızdır. Önsel dağılım $P(\theta)$ olarak ifade edilir. Bayesçiler, veriyi **ön bilgi** ile birleştirir. |
| **Amaç** | **Önsel İnancı Gözlemlere Dayalı Güncellemek 🔄:** Amaç, Bayes Teoremi'ni kullanarak önsel inancı, gözlemlenen verilerle birleştirmek ve daha doğru bir **sonsal (Posterior)** inanç elde etmektir. |
| **Parametreler** | Popülasyon parametreleri **rastgele değişkenler** olarak kabul edilir ve bunlar hakkında bir olasılık dağılımı (inanç) vardır. |
