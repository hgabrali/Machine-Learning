# 🧠 Makine Öğrenimi'nin Olasılık Temelleri (Probability Foundations of Machine Learning)

Makine öğrenimi, büyük ölçüde bilinmeyen bir sonucun, elimizdeki verilere dayanarak ortaya çıkma **olasılığını hesaplama** sanatıdır. Temelde ML, bir olasılık hesaplama makinesi olarak işlev görür.

[JSXGraph](https://jsxgraph.uni-bayreuth.de/home/)

[Bootstrap](https://getbootstrap.com/docs/5.0/getting-started/introduction/)

##### The following books were used as references in this course.

* Probability and Statistics (4th Edition) , Morris H. DeGroot, Mark J. Schervish, Pearson, 2011

* All of Statistics: A Concise Course in Statistical Inference by Larry Wasserman, Springer, 2010 

* Probabilistic Machine Learning: An Introduction by Kevin Patrick Murphy. MIT Press, March 2022.  


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


<img width="1178" height="460" alt="image" src="https://github.com/user-attachments/assets/24a11c98-87af-4091-bf0f-58b1e4fe0ed2" />

# 🧠 Bayesçi İstatistik ve Bernoulli Dağılımı Örneği

Bayesçi İstatistik, olasılığı inanç derecesi olarak ele alan bir çıkarım yöntemidir. Bernoulli dağılımı ise bu yöntemin temellerini anlamak için kullanılan en yaygın ve en basit örneği sağlar.

## 1. Bayesçi İstatistik Temel Kavramları 🎯

Bayesçi çıkarım, parametreleri ($\theta$) **sabit** değerler olarak değil, **rastgele değişkenler** olarak ele alır.

| Bayesçi Terim | Formül | Açıklama |
| :--- | :--- | :--- |
| **Önsel Dağılım (Prior) 💡** | $P(\theta)$ | Veri gözlenmeden önce model parametresi ($\theta$) hakkındaki **başlangıç inancımızdır**. Parametrelerin muhtemel değerlerine ait olasılık dağılımını ifade eder. |
| **Olabilirlik Fonksiyonu (Likelihood) 🔍** | $P(\text{Veri} \mid \theta)$ | Belirli bir $\theta$ parametre değeri verildiğinde, **gözlemlenen veriyi** elde etme olasılığının ne kadar yüksek olduğunu ölçer. Bu, Sıklıkçı istatistiğin de temelini oluşturan veriye dayalı kısımdır. |
| **Sonsal Dağılım (Posterior) 🏆** | $P(\theta \mid \text{Veri})$ | Önsel inancın, yeni gözlemlenen veriler ışığında **güncellenmiş halidir**. Bayes Teoremi kullanılarak elde edilen bu dağılım, parametre hakkındaki nihai inancımızı temsil eder. |
| **Bayes Teoremi** | $P(\theta \mid \text{Veri}) \propto P(\text{Veri} \mid \theta) \cdot P(\theta)$ | Sonsal $\propto$ Olabilirlik $\times$ Önsel demektir. Sonsal dağılımın, Olabilirlik ve Önsel dağılımların çarpımı ile orantılı olduğunu gösterir. |

## 2. Bernoulli Dağılımı (Bernoulli Distribution) 🪙

Bernoulli dağılımı, yalnızca **iki olası sonuçlu** (dichotomous outcome) tek bir rastgele deneyi tanımlar (Örn: Başarı/Başarısızlık, Evet/Hayır, Tura/Yazı).

| Terim | Formül | Açıklama |
| :--- | :--- | :--- |
| **Amaç** | $X \sim \text{Bernoulli}(\theta)$ | Tek bir denemenin sonucunu modellemek. |
| **Parametre** | $\theta$ (veya $p$) | Başarı olasılığıdır. $\theta \in [0, 1]$ aralığındadır. Başarısızlık olasılığı ise $1 - \theta$'dır. |
| **Olasılık Kütle Fonksiyonu (PMF)** | $P(X=x) = \theta^x (1-\theta)^{1-x}$ | $x=1$ (Başarı) için $\theta$, $x=0$ (Başarısızlık) için $1-\theta$ sonucunu verir. |
| **Örnek** | Hileli bir madeni paranın tek bir atışta Tura gelme olasılığı $\theta$'dır. | $X=1$ Tura gelme, $X=0$ Yazı gelme durumunu temsil eder. |

## 3. Bernoulli Örneği Üzerinden Bayesçi Çıkarım (Beta-Binomial Model)

Bayesçi istatistikte, bir madeni paranın hileli olup olmadığını (yani $\theta$ parametresini) tahmin etmek için Bernoulli dağılımı kullanılır.

| Aşama | Uygulama | Terimsel Açıklama |
| :--- | :--- | :--- |
| **1. Önsel Seçimi** | $\theta \sim \text{Beta}(\alpha, \beta)$ | Bernoulli/Binomial olabilirlik fonksiyonu için yaygın olarak **Konjuge Önsel (Conjugate Prior)** olan **Beta Dağılımı** kullanılır. Beta dağılımı, $\theta$ hakkındaki başlangıç inancımızı temsil eder. $\alpha$ ve $\beta$, sırasıyla 'başarı' ve 'başarısızlık' sayısı hakkındaki ön bilgimiz gibi düşünülebilir. |
| **2. Veri (Likelihood) Toplama** | $X = \{x_1, \dots, x_n\}$ | $n$ kez yapılan madeni para atışı verisi toplanır. Toplam başarı sayısı $k = \sum x_i$ olsun. Olabilirlik fonksiyonu, Binomial dağılımın formunu alır: $P(\text{Veri} \mid \theta) = \theta^k (1-\theta)^{n-k}$. |
| **3. Sonsal Hesaplama** | $\theta \mid \text{Veri} \sim \text{Beta}(\alpha', \beta')$ | Bayes Teoremi uygulandığında, Sonsal Dağılım da yine bir **Beta Dağılımı** çıkar. <br>**Güncellenmiş Parametreler:** <br>$\alpha' = \alpha + k$ <br>$\beta' = \beta + (n-k)$ |
| **Sonuç** | Yeni sonsal dağılım, hem eski inancımızı ($\alpha, \beta$) hem de yeni gözlemlenen veriyi ($k, n-k$) birleştirir. $\theta$'nın artık **en olası** değeri, Sonsal Beta Dağılımının modu veya ortalaması olarak alınır. |

<img width="1126" height="502" alt="image" src="https://github.com/user-attachments/assets/d55cd874-50ae-4306-a5a4-932963ef3c4f" />

# 🧠 Bayesçi İstatistik: (Final Summary)

## 1. Bayesçiler Önsel İnançları Günceller (Bayesians update prior beliefs) 🔄

* **Açıklama:** Bayesçi yaklaşımın temel işlevi, **Önsel İnanç (Prior Belief - $P(\theta)$)** olarak adlandırılan başlangıçtaki bilgimizi veya varsayımımızı, gözlemlenen yeni verilerle (**Olabilirlik / Likelihood**) birleştirerek **Sonsal İnanç (Posterior Belief - $P(\theta \mid \text{Veri})$)** elde etmektir.
* **Örnek:** Bir madeni paranın hileli olduğuna inanıyorsunuz ($\theta \approx 0.7$ Tura). Bu, sizin önsel inancınızdır. Parayı 100 kez attınız ve sadece 48 kez tura geldi. Bayes Teoremi, sizin ilk inancınızı bu yeni veriyle birleştirerek $\theta$'nın muhtemelen $0.5$'e daha yakın olduğunu gösteren yeni (sonsal) bir dağılım oluşturur.

---

## 2. Bilgi İçermeyen Önsellerle MAP, MLE ile Aynıdır (MAP with uninformative priors is just MLE) ⚖️

* **MLE (Maksimum Olabilirlik Tahmincisi / Maximum Likelihood Estimator):** Sadece veriyi esas alır ve veriyi en olası kılan parametreyi bulur.
* **MAP (Maksimum Sonsal Tahmini / Maximum A Posteriori):** Bayesçi bir yöntemdir. Veri ve Önsel inancın çarpımını maksimize eder.
* **Bilgi İçermeyen Önsel (Uninformative Prior):** Bu, parametrenin tüm olası değerlerinin eşit olasılığa sahip olduğunu varsayan bir önseldir (Örn: Düzgün Dağılım / Uniform Distribution).
* **Karşılaştırma:** Eğer önsel, tüm olasılıklara eşit ağırlık veriyorsa (yani hiçbir bilgi içermiyorsa), MAP formülünün önsel kısmı sabit bir sayıya döner. (Bu durumda MAP'i maksimize etmek, sadece **Olabilirlik (Likelihood)** kısmını maksimize etmeye eşittir).
* $$\text{Sonsal} \propto \text{Olabilirlik} \times \text{Önsel (Sabit)}$$
* Böylece, MAP tahminleri ($P(\text{Veri} \mid \theta) \cdot P(\theta)$) ile MLE tahminleri ($P(\text{Veri} \mid \theta)$) matematiksel olarak aynı sonucu verir.

---

## 3. Yeterli Veri Olduğunda, MLE ve MAP Tahminleri Yakınsar (With enough data, MLE and MAP estimates usually converge) 📈

* **Açıklama:** Veri setinin büyüklüğü ($n$) çok yüksek olduğunda, veriden gelen bilgi (Olabilirlik) önsel inanca göre çok daha baskın hale gelir.
* **Yakınsama:** Önselin ne kadar güçlü olursa olsun, çok büyük bir veri seti her iki yöntemi de verinin kendisini en iyi açıklayan parametreye doğru iter. Bu nedenle, ML ve İstatistik uygulamalarında, büyük veri setleriyle çalışırken MLE ve MAP tahminleri genellikle **aynı sonuca** ulaşır.

---

## 4. Sınırlı Veri veya Güçlü Önsel İnanç Olduğunda İyidir (Good for instances when you have limited data or strong prior beliefs) 👍

* **Sınırlı Veri:** Sıklıkçı MLE, az veriyle çalışırken aşırı uyum (overfitting) eğilimi gösterir ve güvenilmez sonuçlar verebilir. Bayesçi yaklaşım, bu boşluğu **Önsel Bilgiyle** doldurarak daha mantıklı ve stabilize edilmiş sonuçlar üretir.
* **Güçlü Önsel İnanç:** Eğer elinizde güvenilir bir uzman görüşü, eski deneyler veya ön bilgiler varsa, Bayesçilik bu bilgiyi resmi olarak modele dahil etmenin tek yoludur.

---

## 5. Yanlış Önseller, Yanlış Sonuçlar (Wrong priors, wrong conclusions) 🛑

* **Açıklama:** Bayesçi yaklaşımın temel riskini ve felsefi eleştirisini özetler.
* Eğer bir araştırmacı, konuyla ilgili olmayan, hatalı veya aşırı önyargılı bir önsel seçerse, model bu yanlış inancı kabul eder ve veri ne kadar iyi olursa olsun sonuçlar hatalı ve yanıltıcı olabilir.
* **Örnek:** Bir paranın adil olduğunu ispatlamaya çalışıyorsunuz. Ancak başlangıcı $\theta \approx 0.99$ (neredeyse her zaman tura) gibi yanlış bir önselle belirlerseniz, parayı 100 kez attığınızda bile (50 tura gelse bile), sonsal inanç $0.5$'e yakınsamayacak, inatla $0.9$'un üzerinde kalacaktır.

---

##### Asagidaki tablolarda, Maksimum Olabilirlik Tahmini (Maximum Likelihood Estimation - MLE), Maksimum Sonsal Tahmini (Maximum A Posteriori - MAP) ve Düzenlileştirme (Regularization) kavramlarının Makine Öğreniminde (Machine Learning) nasıl birleştiğini ve bu birleşimin arkasındaki Bayesçi (Bayesian) mantığı açıklamaktadır.

# 💡 Temel İstatistiksel Tahmin Yöntemleri

| Kavram | Ne İşe Yarar? (İşlevi) | Kullanım Zamanı ve Yeri |
| :--- | :--- | :--- |
| **Maksimum Olabilirlik Tahmini (Maximum Likelihood Estimation - MLE) 🎯** | Verilen $\theta$ parametrelerinin, **gözlemlenen veriyi** oluşturma olasılığını ($P(\text{Veri} \mid \theta)$) maksimize eden parametre değerlerini bulur. | Temel olarak **Sıklıkçı (Frequentist)** bir yaklaşımdır. Veri setinin büyük ve önsel (prior) bilgiye gerek duyulmadığı durumlarda, tahminci (estimator) olarak kullanılır. |
| **Maksimum Sonsal Tahmini (Maximum A Posteriori - MAP) 🧠** | Veri ve **Önsel İnanç** ($P(\theta)$) birleştiğinde, $\theta$ parametresinin en olası değerini bulur. Sonsal dağılımın en yüksek noktasıdır. | Temel olarak **Bayesçi (Bayesian)** bir yaklaşımdır. Sınırlı veri olduğunda veya model parametreleri hakkında güçlü bir önsel bilgi olduğunda kullanılır. |
| **Düzenlileştirme (Regularization) ⚖️** | **Kayıp Fonksiyonuna (Loss Function)** bir ceza terimi ekleyerek modelin katsayılarının (coefficients) mutlak değerlerini veya karelerini sınırlar. Modelin karmaşıklığını azaltır. | **Aşırı Uyum (Overfitting)** riskini azaltmak ve modelin genelleştirme (generalization) yeteneğini artırmak için Regresyon ve Sinir Ağları (Neural Networks) gibi birçok ML modelinde kullanılır (Örn: Ridge, Lasso). |

# 🤝 MAP ve Düzenlileştirmenin (Regularization) Birleşimi

## 2. Logaritma Dönüşümü ve Eşitlik 💡

Metnin en kritik kısmı, çarpma (çarpım) işlemini toplama işlemine dönüştürerek (logaritma alarak) Bayesçi formül ile Regresyon formülü arasındaki eşleşmeyi göstermesidir:

### 1. Bayesçi Çıkarım (Sol Taraf):
$$\text{Amacımız} \rightarrow \text{Maksimumlaştırmak}[P(\text{Veri} \mid \text{Model}) \times P(\text{Model})]$$

### 2. Logaritma Alınması:
$$\text{Maksimumlaştırmak}[\log(P(\text{Veri} \mid \text{Model})) + \log(P(\text{Model}))]$$

### 3. Regresyon (Sağ Taraf): Logaritma dönüşümünden çıkan terimler, bilinen Regresyon terimleriyle eşleşir:

| Bayesçi Terim (Logaritmalı) | Regresyon Terimi |
| :--- | :--- |
| $\log(P(\text{Veri} \mid \text{Model}))$ | **Maksimumlaştırmak** $\log(P(\text{Veri} \mid \text{Model}))$ aynı zamanda **Kare Kaybı (Square Loss / Hata Karesi Toplamı)**'nı minimize etmeye eşittir. |
| $\log(P(\text{Model}))$ | **Maksimumlaştırmak** $\log(P(\text{Model}))$), **Katsayıların Karelerinin Toplamını (Sum of Squares of Coefficients)** minimize etmeye eşittir. Bu terim, **Düzenlileştirme Terimi (Regularization Term)** olarak bilinir. |

### 4. Nihai Sonuç: 🚀
* **Yeni Kayıp = Kare Kaybı + Düzenlileştirme Terimi**
* MAP, Kare Kaybını (Hata) minimize etmeyi ve katsayıların karelerinin toplamını (Model Karmaşıklığı) minimize etmeyi birleştirir. Bu, **Ridge Regresyon'un (L2 Düzenlileştirme)** maliyet fonksiyonudur.

---

## 3. P(Model)'in Anlamı (The Probability of a Model) 🧠

Metin, bir modelin olasılığının ($P(\text{Model})$) ne anlama geldiğini açıklıyor:

* **Varsayım:** Modelin katsayılarının ($a_1, a_2, \dots$) **Standart Normal Dağılımdan (Standard Normal Distribution)** seçildiği varsayılır.
* **Hesaplama:** Bir modelin olasılığı, tüm bu katsayıları seçme olasılıklarının çarpımıdır.
* **Basitlik ve Olasılık:** Basit bir modelin (Model 1: 1 katsayı) olasılığı, karmaşık bir modelden (Model 3: 10 katsayı) daha yüksektir, çünkü katsayıların çarpımı daha azdır. Bu da, Bayesçi yaklaşımın, **Basitlik Prensibini (Ockham's Razor)** otomatik olarak model seçimine dahil ettiğini gösterir.

# ✍️ Özet Tablo: Bayesçi ve Regresyon Kavramlarının Eşleştirilmesi

Bu tablo, Bayesçi yaklaşımın (özellikle MAP) logaritma ve eksi işareti dönüşümleri sayesinde Regresyon ve Düzenlileştirme (Regularization) olarak bilinen kavramlarla nasıl matematiksel olarak eşleştiğini göstermektedir.

| Bayesçi Kavram | Matematiksel İşlem | Regresyon Kavramı |
| :--- | :--- | :--- |
| **Maksimumlaştır** $P(\text{Veri} \mid \text{Model})$ | Logaritma Almak ve Eksilisini Almak | **Minimize Et** Kare Kaybı (Square Loss) 📉 |
| **Maksimumlaştır** $P(\text{Model})$ | Logaritma Almak ve Eksilisini Almak | **Minimize Et** Düzenlileştirme Terimi (Regularization Term) 🎚️ |
| **MAP** ($P(\text{Veri} \mid \text{Model}) \cdot P(\text{Model})$) | Logaritma Toplamı | **Minimize Et** Toplam Kayıp Fonksiyonu (Total Loss Function) 💰 |

---
---

### ❓ "Bir Modelin Olasılığı Nedir?" (What Is the Probability of a Model?)

"Bir Modelin Olasılığı Nedir?" sorusu, tek başına kesin bir istatistiksel tanıma sahip değildir. Genellikle bu soru, bağlama göre iki ana yoruma gelir:

1.  **Bayesçi İstatistik Yorumu (Asıl Anlam) 🧠:**
    * **Anlamı:** Modelin, eldeki veriler göz önüne alındığında doğru olma olasılığı nedir?
    * **Kullanım Alanı:** Model Karşılaştırma ve Seçimi.

2.  **Klasik İstatistik Yorumu (Pratik Anlam) 🎯:**
    * **Anlamı:** Modelin bir tahmininin (örneğin bir sınıflandırma modelinin) belirli bir sınıfa ait olma olasılığı nedir?
    * **Kullanım Alanı:** Modelin Çıktısı (Tahmin Güveni).

Bu iki temel yoruma ve ilgili parametrelere detaylı olarak bakmak gereklidir.

| Parametre / Konu | Ne? | Neden Kullanılır? | Nasıl/Niçin Kullanılır? |
| :--- | :--- | :--- | :--- |
| **1. Modelin Posterior Olasılığı ($P(M|D)$) 🧠** | **Bayesçi İstatistik**'te, modelin ($M$), eldeki gözlemlenen veriler ($D$) göz önüne alındığında doğru olma olasılığıdır. | Farklı modelleri (veya model parametrelerini) doğrudan bir olasılıkla karşılaştırmak ve hangi modelin verileri en iyi açıkladığını belirlemek için. | **Bayes Teoremi** kullanılarak hesaplanır: $$P(M|D) = \frac{P(D|M) \cdot P(M)}{P(D)}$$ Burada $P(D|M)$ **Olasılık (Likelihood)**, $P(M)$ **Önsel (Prior)** olasılıktır. |
| **2. Marjinal Olasılık (Model Kanıtı)📊** | **$P(D|M)$ (Likelihood):** Modelin, belirli parametreler altında gözlemlenen verileri üretme olasılığıdır. | Modelin, tahmin edilen parametre değerlerinin ne kadar iyi uyum sağladığını gösterir. | Modeli eğitirken kullanılan veriye uyumu ölçmek ve model parametrelerini optimize etmek için (Maksimum Olabilirlik Tahmini - MLE). |
| **3. Bayes Faktörü (BF) ⚖️** | İki rakip modelin ($M_1$ ve $M_2$), veriler tarafından ne kadar desteklendiğini gösteren oran: $BF = \frac{P(D|M_1)}{P(D|M_2)}$. | İki modelin birbirine karşı destek düzeyini ölçmek ve hangisinin daha olası olduğunu belirlemek için. | Genellikle $\ln(BF)$ logaritması alınarak yorumlanır; $BF > 1$ ise $M_1$ daha olasıdır. |
| **4. Tahmin Olasılığı (Model Çıktısı)🎯** | Bir sınıflandırma modelinin (Lojistik Regresyon, Sınıflandırıcılar vb.) yeni bir veri noktasını belirli bir sınıfa atama olasılığı. | Modelin tahminindeki **belirsizliği** ölçmek ve yalnızca yüksek güvene sahip tahminleri kabul etmek için. | $P(Sınıf|Veri)$. Bu çıktı, genellikle bir Softmax veya Sigmoid aktivasyon fonksiyonu kullanılarak elde edilir. |
| **5. Model Güven Aralığı (Confidence Interval) 🚧** | Bir model parametresinin (örneğin, regresyon katsayısı $\beta$) veya tahmininin gerçek değerini belirli bir güven düzeyinde (örneğin %95) içerme olasılığı. | Model sonuçlarının ne kadar güvenilir veya hassas olduğunu nicel olarak belirtmek için. | Tahmin edilen değer $\pm$ Hata Marjini şeklinde ifade edilir. |
| **6. Model Değerlendirme Metrikleri 📈** | | | |
| a) $R^2$ (Belirleme Katsayısı) | Modelin bağımlı değişkendeki varyasyonu ne ölçüde açıkladığını gösterir. | Regresyon modellerinin genel uyumunu ve açıklama gücünü ölçmek için. | $0$ ile $1$ arasında değişir ($1$ en iyi uyumdur). |
| b) AUC-ROC Eğrisi Altındaki Alan | Modelin rastgele seçilen pozitif bir örneği, rastgele seçilen negatif bir örnekten daha yüksek bir skorla sıralama olasılığıdır. | Sınıflandırma modelinin ayırma gücünü tüm olası eşikler boyunca ölçmek için. | $0.5$ (rastgele tahmin) ile $1.0$ (mükemmel tahmin) arasında değişir. |
| c) P-Değeri | Gözlemlenen etkinin (veya daha aşırı bir etkinin) null hipotezi doğruyken elde edilme olasılığı. | Bir değişkenin (özellik) model üzerinde istatistiksel olarak anlamlı bir etkiye sahip olup olmadığını test etmek için. | Genellikle $p < 0.05$ olduğunda null hipotez reddedilir. |

### 📋 Özetle Cevap Yaklaşımı

Bu soru bir Data Scientist mülakatında sorulduğunda, en doğru ve kapsamlı cevap, iki ana perspektife değinmektir:

1.  **Teorik/Bayesçi Perspektif 🧠:** Soru, genellikle **Bayesçi Model Karşılaştırması** bağlamında **Modelin Posterior Olasılığını ($P(M|D)$)** ifade eder. Bu, modelin kendisinin doğru olma olasılığıdır.

2.  **Pratik/Uygulama Perspektifi ⚙️:** Soru, günlük iş akışında kullanılan bir sınıflandırma modelinin ürettiği **Tahmin Olasılığını** ifade ediyor olabilir.

Bu ayrımları yaparak ve her iki alandaki kritik parametreleri (Posterior Olasılık, Bayes Faktörü ve AUC, $R^2$ gibi pratik metrikler) açıklayarak konuya hakimiyetinizi gösterebilirsiniz.

---

<img width="733" height="205" alt="image" src="https://github.com/user-attachments/assets/d608d7e7-ef6f-46df-9385-29459b775943" />

* Sorunun amacı, popülasyonun temel parametrelerini (varyans, ortalama, oran) tahmin etmek için kullanılan genel istatistiksel yöntemi sormaktadır:

### 🎯 Point Estimation (Nokta Tahmini)

* **Point Estimation (Nokta Tahmini):** Bir popülasyon parametresini (örneğin popülasyon ortalaması $\mu$, varyansı $\sigma^2$ veya oranı $p$) tek bir değerle tahmin etme yöntemidir.
    * **Örneklem Ortalaması ($\bar{x}$)** popülasyon ortalaması ($\mu$) için bir nokta tahminidir.
    * **Örneklem Varyansı ($s^2$)** popülasyon varyansı ($\sigma^2$) için bir nokta tahminidir.
    * **Örneklem Oranı ($\hat{p}$)** popülasyon oranı ($p$) için bir nokta tahminidir.

* Bu nedenle, Nokta Tahmini hem ortalamayı, hem varyansı hem de oranı tahmin etmek için kullanılan genel bir **yöntemdir**.

---

<img width="755" height="260" alt="image" src="https://github.com/user-attachments/assets/373b3968-9db1-4e53-bdd3-ff4b80f5b6b6" />

* 
Bu soru, Maksimum Olabilirlik Tahmini (Maximum Likelihood Estimation - MLE) yöntemini kullanarak bir Bernoulli denemesindeki (madeni para atışı) başarı olasılığını (p) bulmaya ilişkindir.

Sorunun çözümü için izlenmesi gereken adımlar ve kullanılacak fonksiyon şunlardır:

### 🪙 Maksimum Olabilirlik Tahmini (MLE) ile Madeni Para Atışı Sorununun Çözümü

#### 1. Problemi Tanımlama

| Parametre | Değer | Açıklama |
| :--- | :--- | :--- |
| Toplam Deneme Sayısı ($n$) | 10 | (10 kez yazı tura atıldı). |
| Başarı Sayısı ($k$) | 6 | (6 kez tura geldi - "heads"). |
| Başarısızlık Sayısı ($n-k$) | 4 | (4 kez yazı geldi - "tails"). |
| Tahmin Edilecek Parametre | $p$ | Tura gelme olasılığı. |

#### 2. Olabilirlik Fonksiyonu (Likelihood Function) Kurma 📝

Bir madeni para atışı dizisindeki sonuçların olasılığı, Binom Dağılımı kullanılarak hesaplanır. Ancak, MLE'de biz sadece belirli bir dizinin (örneğimizde 6 tura ve 4 yazı) gerçekleşme olasılığını maksimize etmeye odaklanırız.

$p$ tura gelme olasılığı ve $(1-p)$ yazı gelme olasılığı olmak üzere, herhangi bir 6 tura ve 4 yazı dizisinin gerçekleşme olasılığı (olabilirlik fonksiyonu $L(p)$) şu şekilde ifade edilir:

$$L(p) = P(\text{veriler}|p) \propto p^k \cdot (1-p)^{n-k}$$

*Burada $\propto$, Binom olasılık fonksiyonundaki $\binom{n}{k}$ katsayısını (bu katsayı $p$ parametresine bağlı olmadığı için MLE sürecinde genellikle göz ardı edilir) içerdiğini belirtir.*

#### 3. Değerleri Yerine Koyma

Bulduğumuz değerleri fonksiyonda yerine koyarız:

* $k=6$
* $n-k=4$

$$L(p) = p^6 \cdot (1-p)^4$$

#### 4. Maksimum Olabilirlik Tahminini Bulma (Ek Bilgi) 💡

Soruda sadece maksimize edilmesi gereken fonksiyon sorulsa da, tam MLE değeri de bu fonksiyondan türetilir:

* Bu fonksiyonu maksimize eden $p$ değeri, $\frac{d(\ln L(p))}{dp} = 0$ denklemi çözülerek bulunur.
* Bu tür Binom durumlarında, Maksimum Olabilirlik Tahmini her zaman basitçe gözlemlenen oran ($\hat{p}$) olur:

$$\hat{p} = \frac{\text{Başarı Sayısı}}{\text{Toplam Deneme Sayısı}} = \frac{6}{10} = 0.6$$

---
<img width="777" height="235" alt="image" src="https://github.com/user-attachments/assets/f37673c8-ed85-4d46-8edf-ad374e5528d8" />

* Basit Lineer Regresyon modelinin (En Küçük Kareler Yöntemi - Ordinary Least Squares, OLS) temel çalışma prensibini tam olarak açıklamaktadır:
* (Lineer regresyon, noktalar ve uydurulan çizgi arasındaki karesel mesafelerin toplamını minimize ederek veriye en iyi uyumu sağlar.)
  
---

<img width="724" height="225" alt="image" src="https://github.com/user-attachments/assets/10ef8e7c-81bd-401d-891c-d34b45ba484e" />

### 🛡️ Düzenlileştirme (Regularization) Amacı

**Doğru İfade:**
> Regularization prevents overfitting by penalizing models with large coefficients or weights.
> (Düzenlileştirme, büyük katsayılara veya ağırlıklara sahip modelleri cezalandırarak aşırı uydurmayı (overfitting) önler.)

#### Neden Bu Tanım Doğrudur?

Düzenlileştirmenin temel amacı ve mekanizması şunlardır:

* **Birincil Amaç 🎯:** Düzenlileştirmenin (**L1 - Lasso** veya **L2 - Ridge** gibi) asıl amacı, modelin **karmaşıklığını** kontrol altına alarak eğitim verisine aşırı derecede uyum sağlamasını (**overfitting**) engellemektir.
* **Mekanizma ⚖️:** Bu, kayıp fonksiyonuna (**loss function**) katsayıların (ağırlıkların) büyüklüğüne bağlı bir **ceza terimi (penalty term)** eklenerek yapılır.
* **Etkisi 📉:** Büyük katsayılar, modelin verideki küçük dalgalanmalara karşı çok hassas olduğu anlamına gelir. Bu katsayıları cezalandırmak, modeli daha **genelleştirilebilir** (basit) hale getirir.

---

<img width="768" height="229" alt="image" src="https://github.com/user-attachments/assets/7a35c4b7-8958-4c9d-8c49-a51c50b44ad8" />

### 📏 L2 Düzenlileştirme Hata Değeri Hesaplaması (Ridge Penalty)

Soruda istenen, verilen $M = 4x^4 + 3x^2 + 1$ modeli için **L2 Düzenlileştirme Hata Değeri**'ni (L2 regularization error value) hesaplamaktır.

L2 düzenlileştirme (Ridge Regresyonu), modelin katsayılarının (ağırlıklarının) karesinin toplamına eşittir. **Bias terimi** (kesişim/sabit terim) genellikle düzenlileştirmeye dahil edilmez.

#### 1. Modelin Katsayılarını (Ağırlıklarını) Belirleme:

Verilen model:
$$M = 4x^4 + 3x^2 + 1$$

Bu modelin katsayıları (ağırlıkları) şunlardır:
* $x^4$ teriminin katsayısı ($w_4$): $4$
* $x^2$ teriminin katsayısı ($w_2$): $3$
* Sabit terim/Bias ($b$): $1$

#### 2. L2 Düzenlileştirme Hata Değeri Formülü:

L2 cezası (penalty) katsayıların karesinin toplamıdır:
$$\text{L2 Cezası} = \sum_{i} w_i^2$$

#### 3. Hesaplama 🔢:

Bias terimini ($1$) hariç tutarak katsayıların karesini toplarız:

$$\text{L2 Cezası} = (4)^2 + (3)^2$$
$$\text{L2 Cezası} = 16 + 9$$
$$\text{L2 Cezası} = 25$$

---

**Sonuç:** L2 düzenlileştirme hata değeri **25**'tir.

**Not 📌:** Bu değer, toplam kayıp fonksiyonuna ($\text{Loss}$) $\lambda$ (lambda) hiperparametresi ile çarpılarak eklenir: $\text{Toplam Loss} = \text{MSE} + \lambda \cdot (\text{L2 Cezası})$. Ancak soruda sadece ceza değerinin kendisi (25) istenmiştir.

---

### 🔎 Maksimum Olabilirlik ile Dağılım Karşılaştırması

<img width="821" height="616" alt="image" src="https://github.com/user-attachments/assets/e2d0af40-ec6e-479d-8d23-401e98490320" />


Bu soru, $S = \{-1, 2\}$ örneklemini üretme olasılığı en yüksek olan dağılımı bulmak için **Maksimum Olabilirlik (Maximum Likelihood - ML)** ilkesini kullanmayı gerektirir.

Örneklemdeki gözlemler bağımsız kabul edildiğinden, bir dağılımın ($M$) Olabilirlik Fonksiyonu (L(M)), şu şekilde hesaplanır:

$$L(M) = f(x_1 | M) \cdot f(x_2 | M)$$

$f(x|M)$ değeri, grafikteki **Yoğunluk (Density)** değerleridir.

---

#### 1. Dağılım $N(0, 2^2)$ İçin Olabilirlik Hesaplaması 🟢

Bu dağılım için $\mu=0$ ve $\sigma=2$'dir.

| Nokta ($x$) | Yoğunluk $f(x)$ Değeri |
| :---: | :---: |
| $x_1 = -1$ | $0.18$ |
| $x_2 = 2$ | $0.12$ |

$$\text{L}(N(0, 2^2)) = 0.18 \cdot 0.12 = 0.0216$$

#### 2. Dağılım $N(1, 1^2)$ İçin Olabilirlik Hesaplaması 🔴

Bu dağılım için $\mu=1$ ve $\sigma=1$'dir.

| Nokta ($x$) | Yoğunluk $f(x)$ Değeri |
| :---: | :---: |
| $x_1 = -1$ | $0.05$ |
| $x_2 = 2$ | $0.24$ |

$$\text{L}(N(1, 1^2)) = 0.05 \cdot 0.24 = 0.0120$$

---

#### 3. Karşılaştırma Sonucu

| Dağılım | Olabilirlik Değeri ($L$) |
| :--- | :--- |
| $N(0, 2^2)$ | **$0.0216$** |
| $N(1, 1^2)$ | $0.0120$ |

$0.0216 > 0.0120$ olduğundan, **$N(0, 2^2)$ dağılımının** verilen örneklemi üretme olasılığı (olabilirliği) daha yüksektir.

**Doğru Cevap: $N(0, 2^2)$**

---

<img width="734" height="239" alt="image" src="https://github.com/user-attachments/assets/72cc4f4d-fa93-427d-9b9f-ae1406a500f4" />

### 💡 Önsel (Prior) İnançların Bayes İstatistiğindeki Rolü

Kavramsal olarak "önsel" (prior) terimi, herhangi bir veri gözlemlemeden önce bir parametrenin dağılımı hakkındaki **başlangıç inançlarını** yansıtır.

Ancak, Bayes istatistiğinde, bu önsel inançların gücü, gözlemlenen veriler tarafından yönlendirilen sistematik bir **güncelleme süreci** aracılığıyla değiştirilir:

* **Güncelleme Mekanizması 🔄:** Bu güncelleme, Bayes teoremi aracılığıyla yapılır. Bu süreç, önsel inançların, verilerden elde edilen bilgilerle rafine edilmesini ve hizalanmasını sağlar.
* **Verinin Merkezi Rolü 📊:** Bayes analizinde **veri asla göz ardı edilmez veya dışlanmaz**. Veri, temel gerçekliği daha iyi yansıtmak üzere önsel dağılımı ayarlamada ve şekillendirmede merkezi bir rol oynar.

---

<img width="834" height="600" alt="image" src="https://github.com/user-attachments/assets/975ba114-e275-4f23-82c7-435517229ad4" />

### ⚖️ MAP (Maximum A Posteriori) Tahmini ve Önsel Karşılaştırması

Bu çözüm, farklı **Önsel (Prior)** inançlara sahip iki Bayesçi'nin, aynı veriyi gözlemledikten sonra ulaştıkları **Maksimum Ardsıl Olasılık (MAP)** tahminlerini karşılaştırmaktadır.

#### 1. Veri ve Modeli Tanımlama 📊

| Parametre | Değer |
| :--- | :--- |
| Deneme Sayısı ($n$) | 10 |
| Tura Sayısı (Başarı, $k$) | 3 |
| Yazı Sayısı (Başarısızlık, $n-k$) | 7 |
| Tahmin Edilecek Parametre | Tura gelme olasılığı ($\theta = P(H)$) |

**Olabilirlik (Likelihood) Fonksiyonu:** $\propto \theta^3 (1-\theta)^7$

---

#### 2. Bayesçi 2'nin MAP Tahmini (Uniform/Zayıf Önsel)

Bayesçi 2, **uniform önsel** ($\text{Prior} \propto 1$) kullanır. Bu, tüm olasılıklara eşit ağırlık verdiği için bilgi içermez.

$$\text{Posterior} \propto \theta^3 (1-\theta)^7 \cdot 1$$

* Bu durumda MAP tahmini, **Maksimum Olabilirlik Tahmini (MLE)** ile aynıdır:
    $$\text{MAP}_{\text{Bayesçi 2}} = \frac{\text{Başarı Sayısı}}{\text{Toplam Deneme Sayısı}} = \frac{3}{10} = \mathbf{0.30}$$

---

#### 3. Bayesçi 1'in MAP Tahmini (Güçlü Önsel)

Bayesçi 1, paranın adil ($\mathbf{P(H)=0.5}$) olduğuna dair **güçlü bir önsel** inanç kullanır. Bu güçlü önsel, zayıf veriye rağmen sonucu kendi ortalamasına yaklaştırır.

$$\text{MAP}_{\text{Bayesçi 1}} \text{ değeri } \in [0.30, 0.50]$$

* **Prensip 🎯:** Güçlü bir önsel, zayıf bir veriyi tamamen ezemez, ancak tahminin veri oranı ($0.30$) ile önselin ortalaması ($0.50$) arasında kalmasını sağlar. Önsel güçlü olduğu için, sonuç **$0.50$'ye daha yakın** olacaktır.
* **Makul Değerler:** $0.30$ ve $0.50$ arasındaki $0.49$ veya $0.51$ seçenekleri, güçlü önselin etkisini yansıtır.

**Beklenen Sonuç:**
* Bayesçi 1: Güçlü önsel nedeniyle $0.50$'ye yakın (örn., $\mathbf{0.49}$)
* Bayesçi 2: Veri oranı olduğu için $\mathbf{0.30}$

---

### 4. Seçeneklerin Değerlendirilmesi (Bayesçi MAP Karşılaştırması) 🧠

| Seçenek | Bayesçi 1 (Güçlü Önsel, 0.5 civarı) | Bayesçi 2 (Zayıf Önsel, 0.3 civarı) | Karar |
| :---: | :---: | :---: | :--- |
| **A** | $P(H) = 0.49$ | $P(H) = 0.30$ | **Mantıklı** (Bayesçi 1, 0.30 ile 0.50 arasını, 0.50'ye yakın seçmiştir.) |
| **B** | $P(H) = 0.51$ | $P(H) = 0.30$ | Mantıklı (Bayesçi 1, 0.50'nin biraz üzerine kaymıştır.) |
| **C** | $P(H) = 0.30$ | $P(H) = 0.30$ | Yanlış (Bayesçi 1'in güçlü önseli yok sayılmıştır.) |
| **D** | $P(H) = 0.30$ | $P(H) = 0.49$ | Yanlış (Bayesçi 1'in ve Bayesçi 2'nin sonuçları karıştırılmıştır.) |

**Not:** Bu karşılaştırmada, Bayesçi 2'nin uniform önseli nedeniyle MAP değeri doğrudan gözlemlenen orana (0.30) eşit çıkarken; Bayesçi 1'in güçlü önseli, sonucu 0.50'ye yakın tutmaktadır.

### 🌟 Bayesçi Sonuçların Yorumlanması (MAP Kararı)

Bayesçi modellemede, güçlü bir önselin zayıf bir veri kümesiyle karşılaştığı durumlar önemlidir:

* Hem **$0.49$** hem de **$0.51$** mantıklıdır, ancak Bayesçi modelleme genellikle sonuçları önselin ortalamasına yakın tutar.
* Bu tür çoktan seçmeli sorularda, genellikle güçlü önselin $0.50$'den sadece biraz uzaklaştığı kabul edilir.
* Gözlemlenen Veri ($0.30$) ve Önsel ($0.50$) arasında, $0.50$'ye en yakın olan değerler $0.49$ veya $0.51$'dir.

Deneyimli Bayesçilerin çoğu, güçlü bir önselin $10$ atışlık zayıf bir veriyi tamamen ezemeyeceğini bilir, bu yüzden $0.50$'nin hemen yanındaki $0.49$ veya $0.51$ en olası değerlerdir.

**Bu bağlamda, ilk seçenek (A) en tipik Bayesçi sonucu yansıtır:**

| Bayesçi | Önsel Tipi | Sonuç Yorumu | MAP Değeri |
| :---: | :---: | :--- | :---: |
| **Bayesçi 1** | Güçlü Önsel | $\text{Güçlü Önsel} + \text{Zayıf Veri} \implies$ Önsele yakın | $\mathbf{0.49}$ |
| **Bayesçi 2** | Uniform Önsel | $\text{Uniform Önsel} + \text{Zayıf Veri} \implies$ Veriye eşit | $\mathbf{0.30}$ |

**Doğru cevap ilk seçenektir:** **Bayesian 1: $P(H)=0.49$, Bayesian 2: $P(H)=0.30$.**

---
---
---

### 📊 Nicel İstatistik: Güven Aralığı (Confidence Interval) Kavramları

* Görseldeki maddeler, nicel istatistiğin temel konularından biri olan Güven Aralığı (Confidence Interval) kavramını ve bu aralığı etkileyen faktörleri açıklamaktadır.
  
Güven aralığı, bir popülasyon parametresini (örneğin ortalama $\mu$) tahmin etmek için kullanılan, örneklem verisine dayalı istatistiksel bir aralıktır.

| Kavram | Açıklama | Anahtar Çıkarım | Emoji |
| :--- | :--- | :--- | :---: |
| **Confidence Intervals** (Güven Aralıkları) | Güven aralıkları, örneklem ortalamasına ($\bar{x}$) her iki taraftan da bir hata payının eklenmesiyle elde edilir. $$\text{GA} = \bar{x} \pm \text{Hata Payı}$$ | Popülasyon parametresi için tahmini bir aralık sunar. | 📏 |
| **Confidence Level** (Güven Seviyesi) | Bir güven aralığının, tahmin edilen gerçek popülasyon ortalamasını ($\mu$) içerme olasılığıdır (örneğin %95). | Güven aralığının gerçek parametreyi yakalama olasılığını gösterir. | 🎯 |
| **Ideally** (İdeal Durum) | İdeal olarak, hem yüksek güven seviyesine hem de dar bir aralığa sahip olmak istenir. (Yüksek kesinlik + Yüksek doğruluk) | Yüksek güven ve dar aralık bir çelişkidir; bu hedefe ancak daha fazla veriyle ulaşılabilir. | ✨ |
| **Larger Samples** (Daha Büyük Örneklemler) | Daha büyük örneklemler (daha fazla veri), hata payını düşürerek daha dar bir aralık sağlayacaktır. | Dar aralık ve yüksek güven elde etmenin en güvenilir yoludur. | 📈 |
| **Decreasing Confidence Level** (Güven Seviyesini Düşürmek) | Güven seviyesini düşürmek (%99'dan %95'e gibi) de aralığı daraltacaktır. | Aralığı daraltır (kesinliği artırır) ancak gerçek ortalamayı yakalama olasılığını (güveni) azaltır. | 📉 |

<img width="1213" height="590" alt="image" src="https://github.com/user-attachments/assets/11d81451-7fa5-4ee5-966c-caa7cb609d9f" />

# 📈 Hata Payı (Margin of Error) Tahmini ve İstatistiksel Temeller

Bu bölümdeki görsel, **Hata Payı (Margin of Error)** kavramını ve bir popülasyonun **Normal Dağılım** özelliğini ($X \sim N(\mu, \sigma^2)$) göstermektedir. Hata Payı'nı tahmin etmek, yani bir **Güven Aralığı** oluşturmak için kullanılan formül, elinizdeki istatistiksel bilgiye ve örneklem büyüklüğüne bağlı olarak değişir.

---

## Temel Formül ve Bileşenler

Hata Payı'nın temel formülü şöyledir:

$$\text{Hata Payı} = \text{Kritik Değer} \times \text{Standart Hata}$$

| Bileşen | Açıklama |
| :--- | :--- |
| **Kritik Değer** ($\boldsymbol{Z}$ veya $\boldsymbol{t}$) | Seçtiğiniz Güven Seviyesine (örneğin, %95) göre Normal Dağılım ($Z$) veya t-Dağılım ($t$) tablosundan belirlenen değerdir. |
| **Standart Hata** | Örneklem ortalamasının, popülasyon ortalamasından ne kadar uzakta olabileceğinin bir ölçüsüdür. |

---

## 🔬 Tahmin Durumları

Görselde popülasyonun Normal dağılıma sahip olduğu belirtildiği için (bu, nicel istatistikte sıklıkla varsayılır), tahmin formülü şu iki ana duruma göre belirlenir:

### Durum 1: Popülasyon Standart Sapması ($\sigma$) **BİLİNİYORSA**

Bu durumda (pratikte nadir) Hata Payı'nı hesaplamak için **Z-Skoru (Normal Dağılım)** kullanılır.

$$\text{Hata Payı} = Z_{\alpha/2} \times \frac{\sigma}{\sqrt{n}}$$

| Parametre | Açıklama | Örnek Değerler |
| :--- | :--- | :--- |
| $\boldsymbol{Z_{\alpha/2}}$ (Kritik Değer) | Seçilen güven seviyesine karşılık gelen $z$-skoru. | %95 Güven Seviyesi için $Z_{\alpha/2} \approx 1.96$ |
| $\boldsymbol{\sigma}$ | Popülasyon Standart Sapması (Görselde $\sigma^2$ (varyans) ile gösterilen değerin karekökü). | |
| $\boldsymbol{n}$ | Örneklem Büyüklüğü (Toplanan veri sayısı). | |
| $\boldsymbol{\frac{\sigma}{\sqrt{n}}}$ | Standart Hata (Örneklem ortalamasının standart sapması). | |

---

### Durum 2: Popülasyon Standart Sapması ($\sigma$) **BİLİNMİYORSA**

Pratikte en yaygın durum budur. Bu durumda, popülasyon standart sapması yerine **örneklem standart sapması ($s$)** kullanılır ve genellikle **t-Dağılımı** kullanılır.

$$\text{Hata Payı} = t_{n-1, \alpha/2} \times \frac{s}{\sqrt{n}}$$

| Parametre | Açıklama |
| :--- | :--- |
| $\boldsymbol{t_{n-1, \alpha/2}}$ (Kritik Değer) | Seçilen güven seviyesine ve serbestlik derecesine ($n-1$) karşılık gelen $t$-skoru. |
| $\boldsymbol{s}$ | Örneklem Standart Sapması (Topladığınız örneklem verisinin standart sapması). |
| $\boldsymbol{n}$ | Örneklem Büyüklüğü (Toplanan veri sayısı). |

### ⚠️ Önemli Not (Merkezi Limit Teoremi)

Eğer örneklem büyüklüğü ($n$) yeterince büyükse (genellikle $n \ge 30$ kabul edilir), **Merkezi Limit Teoremi** gereği popülasyonun dağılımı ne olursa olsun, örneklem ortalamasının dağılımı normale yaklaşır ve bu durumda $t$-Dağılımı yerine yine yaklaşık olarak $Z$-Dağılımı kullanılabilir.

---

## 🤖 Makine Öğrenmesi Bağlamında Önemi

Hata Payı ve Güven Aralığı kavramı, makine öğrenmesi ve veri analizinde özellikle kritik noktalarda devreye girer:

1.  **Regresyon Analizi (Tahminler):**
    * Tahmin edilen katsayılar için güven aralıkları hesaplanır. Bu, katsayının gerçek değerinin büyük olasılıkla hangi aralıkta olduğunu gösterir.

2.  **A/B Testi (Karşılaştırma):**
    * İki grup arasındaki farkın istatistiksel olarak anlamlı olup olmadığını test ederken, fark için bir güven aralığı oluşturulur. Eğer bu aralık sıfırı içermiyorsa, farkın anlamlı olduğu sonucuna varılır.

3.  **Model Değerlendirme (Güvenilirlik):**
    * Modelinizin performans ölçümlerine (örneğin Doğruluk - Accuracy) Güven Aralığı eklemek, tahminlerinizin ne kadar güvenilir olduğunu daha şeffaf bir şekilde gösterir.
  
<img width="1183" height="560" alt="image" src="https://github.com/user-attachments/assets/811d0dac-3743-4465-9eb3-65a637c0db8d" />

# ⚖️ Veri Dağılımları Üzerine: Merkezi Limit Teoremi (CLT)

Bu görsel, istatistikteki en önemli kavram olan **Örneklem Ortalamasının Dağılımı (Sampling Distribution of the Mean)** ve **Merkezi Limit Teoremi'ni (CLT)** açıklamaktadır.

## Temel Kural

Bir popülasyondan alınan örneklem ortalamalarının dağılımı, her zaman popülasyon ortalaması ($\mu$) etrafında merkezlenir ve varyansı popülasyon varyansının ($\sigma^2$) örneklem büyüklüğüne ($n$) bölünmesiyle bulunur: $\frac{\sigma^2}{n}$.

| Senaryo | Popülasyon (Ana Kitle) Dağılımı | Uygulama Kuralı | Örneklem Ortalamasının Dağılımı ($\bar{X}$) | Anlamı |
| :---: | :--- | :--- | :--- | :--- |
| **Senaryo 1** 🎯 | **Normal Dağılım** ($\boldsymbol{X \sim N(\mu, \sigma^2)}$) | **Doğrudan Kural:** Popülasyon Normal olduğu için, örneklem dağılımı her zaman normaldir. | **Normal Dağılım:** $$\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right)$$ | Örneklem büyüklüğü ($n$) ne olursa olsun, istatistiksel çıkarım için Normal Dağılım kuralları uygulanabilir. |
| **Senaryo 2** 💡 | **Bilinmiyor veya Normal Değil** ($\boldsymbol{X}$ unknown or not Normal) | **Merkezi Limit Teoremi (CLT):** Yeterince büyük bir örneklem alınmalıdır ($n \ge 30$). | **Yaklaşık Normal Dağılım:** $$\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right)$$ | Popülasyonun şekli önemli değildir. Büyük örneklemler sayesinde, istatistiksel analizler için Normal Dağılımın güçlü matematiksel araçlarını kullanabiliriz. |

## 🌟 Merkezi Limit Teoremi (CLT) Nedir?

CLT, **veri biliminin ve çıkarımsal istatistiğin bel kemiğidir.**

* **Tanım:** Bir popülasyonun dağılımı ne olursa olsun, o popülasyondan alınan rastgele ve yeterince büyük örneklemlerin ortalamalarının dağılımı, yaklaşık olarak Normal Dağılıma uyar.
* **Pratikteki Önemi:** Veri bilimcileri ve analistler, çoğu zaman popülasyonun tamamını bilemezler veya ölçemezler. CLT sayesinde, sadece **büyük bir örneklem** toplayarak, popülasyon hakkında güvenilir istatistiksel tahminler yapabilirler.

<img width="1205" height="589" alt="image" src="https://github.com/user-attachments/assets/9564ba7e-eb04-4435-9720-a81e5a77e4b9" />

# 🎯 Güven Aralığı (Confidence Interval) - Hesaplama Adımları

Bu görsel, popülasyon standart sapmasının ($\sigma$) bilindiği veya büyük örneklemle (Z-Dağılımı) çalışıldığı durumda Güven Aralığı hesaplamasını özetler.

## 📝 Hesaplama Adımları (STEPS)

Güven Aralığı: $\text{GA} = \bar{x} \pm Z_{1-\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$

| Adım # | Açıklama | Amacı ve Rolü | Emoji |
| :---: | :--- | :--- | :---: |
| **1** | **Find the sample mean ($\bar{x}$)** | Örneklem ortalamasını bulun. Bu, güven aralığının merkezini oluşturur. | 📋 |
| **2** | **Define a desired confidence level ($\mathbf{1-\alpha}$)** | İstenen güven seviyesini (%95 gibi) tanımlayın. Kritik değeri belirlemeye yarar. | ✅ |
| **3** | **Get the critical value ($\mathbf{Z_{1-\alpha/2}}$)** | Seçilen güven seviyesine karşılık gelen Z-skorunu bulun (Örn: %95 için $\pm 1.96$). | 📐 |
| **4** | **Find the standard error ($\mathbf{\frac{\sigma}{\sqrt{n}}}$)** | Standart hatayı hesaplayın. Bu, örneklem ortalamalarının yayılımını ölçer. | 🌐 |
| **5** | **Find the margin of error (Hata Payını Bulun)** | Kritik değer ile Standart Hatayı çarpın: $$\text{Hata Payı} = Z_{1-\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$ | 🧭 |
| **6** | **Add/subtract the margin of error to the sample mean** | Hata payını örneklem ortalamasına ($\bar{x}$) ekleyip çıkarın. Bu, aralığın alt ve üst limitlerini belirler. | ➕➖ |

## 📊 Güven Aralığı Formülü ve Bileşenleri

| Bileşen | Formül/Gösterim | Açıklama |
| :--- | :--- | :--- |
| **Güven Aralığı** | $$\bar{x} \pm \text{Hata Payı}$$ | Popülasyon ortalamasının büyük ihtimalle içinde bulunduğu aralık. |
| **Hata Payı (Margin of Error)** | $$Z_{1-\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$ | Örneklem ortalamasının, gerçek popülasyon ortalamasından sapabileceği maksimum değer. |
| **Kritik Değer** | $$Z_{1-\alpha/2}$$ | Güven seviyesini temsil eden Z-Skoru (Örn: 1.96). |
| **Standart Hata** | $$\frac{\sigma}{\sqrt{n}}$$ | Örneklem ortalamalarının standart sapması. |

# ⚙️ İstatistiksel Çıkarım Varsayımları (Assumptions)

Bu varsayımlar, Güven Aralığı ve Hipotez Testleri gibi çıkarımsal istatistik yöntemlerinin geçerli ve güvenilir olması için kritik öneme sahiptir.

| Varsayım | Açıklama | Neden Önemli? | Emoji |
| :--- | :--- | :--- | :---: |
| **Simple random sample** (Basit Rastgele Örneklem) | Örneklem, popülasyondaki her bireyin eşit seçilme şansına sahip olduğu bir yöntemle toplanmalıdır. | 🛡️ **Tarafsızlık ve Bağımsızlık:** Örneklemin popülasyonu tarafsız (önyargısız) temsil etmesini ve gözlemlerin birbirinden bağımsız olmasını sağlar. | 🎲 |
| **Sample size > 30 OR** **Population is approximately normal** | **(VEYA)** Örneklem büyüklüğü $n > 30$'dan büyük olmalıdır **ya da** popülasyonun kendisi yaklaşık olarak Normal dağılıma sahip olmalıdır. | 🔔 **Merkezi Limit Teoremi (CLT):** Bu koşullardan birinin sağlanması, örneklem ortalamalarının dağılımının (yaklaşık olarak) Normal Dağılıma uymasını garanti eder. İstatistiksel testlerin temelini oluşturur. | 🧠 |

---


<img width="1197" height="586" alt="image" src="https://github.com/user-attachments/assets/4e1c9de9-af26-4cba-b240-2f1f8c5a30cd" />

* Bu görsel, istatistikteki en yaygın yanlış anlaşılmalardan biri olan Güven (Confidence) ve Olasılık (Probability) arasındaki farkı mükemmel bir şekilde özetlemektedir.

# 🎯 Güven (Confidence) ve Olasılık (Probability) Arasındaki Fark

Bu ayrım, istatistikteki Güven Aralığı kavramının doğru anlaşılması için kritik öneme sahiptir. Temel fark: **Parametre Sabit, Aralık Değişken.**

---

## 1. Doğru Yorumlama (Güven Seviyesi) - ✅

**İfade:** The confidence interval contains the true population parameter approximately 95% of the time.
**(Türkçe: Güven aralığı, gerçek popülasyon parametresini yaklaşık olarak zamanın %95'inde içerir.)**

### ✍️ Açıklama

Güven seviyesi, **yöntemin güvenilirliğini** gösterir. Bir güven aralığı hesaplama sürecini (örneklem alma, aralık hesaplama) sonsuz kez tekrarladığınızı varsayın. Oluşturduğunuz aralıkların yaklaşık %95'i, gerçekten doğru popülasyon parametresini (örneğin $\mu$) yakalayacaktır. **Güven, bu tekrarlanan süreçten gelir.**

### 💰 Örnek

* Bir şirketteki çalışanların gerçek ortalama maaşını ($X$ TL) tahmin etmek istiyorsunuz.
* Her gün 100 farklı çalışan grubundan bir örneklem alıp, 100 farklı %95 güven aralığı hesaplıyorsunuz.
* Bu 100 aralıktan yaklaşık **95 tanesi** gerçek ortalama maaş olan $X$ değerini içerecektir. Geri kalan 5 tanesi ise içermeyecektir.

---

## 2. Yanlış Yorumlama (Olasılık) - ❌

**İfade:** There's a 95% probability that the population parameter falls within the confidence interval.
**(Türkçe: Popülasyon parametresinin güven aralığı içine düşme olasılığı %95'tir.)**

### ✍️ Açıklama

Tek bir güven aralığı hesaplandıktan ve sınırları belirlendikten sonra, gerçek popülasyon parametresi **ya o aralığın içindedir (Olasılık 1) ya da dışındadır (Olasılık 0).** Parametrenin aralığa düşme olasılığı %95 olamaz, çünkü **parametre sabittir.** Sizin yaptığınız şey, doğru sonucu yakalayan bir aralık oluşturma yöntemine %95 oranında güvenmektir.

### 🛑 Örnek

* Sadece **bir kez** %95 güven aralığı hesapladınız ve aralık $[4000 \text{ TL}, 4500 \text{ TL}]$ çıktı.
* Gerçek ortalama maaş ($\mu$) **ya bu aralığın içindedir ya da dışındadır.** $\mu$'nun bu aralığın içine düşme olasılığı %95 **değildir.**

---

**Kısacası: Parametre sabit, aralık değişkendir.**

# ⚖️ Güven (Confidence) ve Olasılık (Probability) Arasındaki Fark

Bu görsel, istatistikteki %95 Güven Seviyesinin (Confidence Level) doğru ve yanlış yorumlanma şekillerini göstererek temel bir kavram hatasını düzeltmektedir.

## Temel İlke

İstatistiksel çıkarımda **popülasyon parametresi (örneğin $\mu$) sabittir**, **güven aralığı ise örneğe göre değişen değişkendir**. Bu nedenle, tek bir aralık için olasılıktan bahsedilemez.

| Durum | İfade | Yorumlama (Neden Doğru/Yanlış?) | Emoji |
| :---: | :--- | :--- | :---: |
| **DOĞRU** ✅ | **The confidence interval contains the true population parameter approximately 95% of the time.** (Güven aralığı, gerçek popülasyon parametresini yaklaşık olarak zamanın %95'inde içerir.) | **✅ Yönteme Güven:** Bu ifade, **tekrarlanan deneme sürecinin** başarı oranını belirtir. Yöntem, oluşturulan aralıkların %95'inin gerçek parametreyi yakalaması için tasarlanmıştır. | 🎯 |
| **YANLIŞ** ❌ | **There's a 95% probability that the population parameter falls within the confidence interval.** (Popülasyon parametresinin güven aralığı içine düşme olasılığı %95'tir.) | **❌ Tekil Olasılık:** Tek bir aralık hesaplandıktan sonra, gerçek parametre ya aralığın içindedir (Olasılık 1) ya da dışındadır (Olasılık 0). Parametre sabit olduğu için, onun aralığa düşme olasılığından söz etmek hatalıdır. | 🛑 |

## Özet

Güven Seviyesi, **"Bu aralığı oluşturan yönteme %95 oranında güveniyorum"** demektir. Olasılık ise **tek bir olayın** gerçekleşme şansıdır.

# 🤖 Makine Öğrenmesi (ML) Regresyon Metrikleri

Bu metrikler, bir regresyon modelinin tahminlerinin, gerçek değerlere ne kadar yakın olduğunu (model hatasını) ölçmek için kullanılır.

| Metrik (Kısaltma) | İngilizce Adı | Açılımı ve Hesaplama Yöntemi | ML'deki Önemi ve Kullanımı | Emoji |
| :---: | :--- | :--- | :--- | :---: |
| **MSE** | **Mean Squared Error** | **Ortalama Karesel Hata.** Hata (gerçek değer - tahmin) hesaplanır, karesi alınır ve tüm örnekler için ortalaması bulunur. Büyük hataları cezalandırır. | ⚡ **Kullanım:** Türevlenebilir olması nedeniyle Gradyan İniş (Gradient Descent) gibi optimizasyon algoritmalarında yaygın olarak kayıp fonksiyonu (loss function) olarak kullanılır. **Dezavantaj:** Birimi, hedef değişkenin biriminin karesidir. |  squared |
| **MAE** | **Mean Absolute Error** | **Ortalama Mutlak Hata.** Hatanın mutlak değeri alınır ve tüm örnekler için ortalaması bulunur. | 🛡️ **Kullanım:** Birim ile aynı birimde olduğu için sonuçların yorumlanması kolaydır. Özellikle aykırı değerlerin (outliers) cezalandırılmasının istenmediği durumlarda tercih edilir. **Dezavantaj:** Mutlak değer fonksiyonu nedeniyle türevi her yerde yoktur, optimizasyonda zorluk çıkarabilir. | 🔢 |
| **MAPE** | **Mean Absolute Percentage Error** | **Ortalama Mutlak Yüzde Hata.** Mutlak hata, gerçek değere bölünerek yüzdeye çevrilir ve ortalaması alınır. | เปอร์เซ็นต์ **Kullanım:** Tahmin doğruluğunu yüzde cinsinden ifade ettiği için iş birimleri ve yönetim tarafından kolayca anlaşılır. **Dezavantaj:** Gerçek değer sıfıra yakın olduğunda tanımsız olabilir veya çok büyük değerler alabilir. | 💰 |
| **Max Error** | **Maximum Error** | **Maksimum Hata.** Tahmin edilen değer ile gerçek değer arasındaki mutlak farkın, veri setindeki en büyük değeridir. | 🚨 **Kullanım:** Modelin **en kötü** durumda ne kadar yanlış tahmin yaptığını gösterir. Özellikle güvenlik ve kritik sistemlerdeki en büyük riskin ölçülmesi için önemlidir. | ❌ |
| **R-squared** | **Coefficient of Determination** | **Belirlilik Katsayısı.** Modelin, hedef değişkendeki varyasyonun ne kadarını açıkladığını gösterir. 0 ile 1 arasında değer alır (bazı durumlarda negatif olabilir). | ✨ **Kullanım:** Modelin ne kadar iyi oturduğunu ve ne kadar açıklayıcı olduğunu anlamak için en popüler metriktir. Yüksek değerler (1'e yakın), modelin veriyi iyi açıkladığını gösterir. **Dikkat:** Fazla parametre eklendiğinde R-squared her zaman artar, bu nedenle *Adjusted R-squared* (Düzeltilmiş R-squared) tercih edilebilir. | 🔍 |

--

