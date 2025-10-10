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
