# 🚀 Machine Learning Temelleri: Yaşam Döngüsü, Veri Türleri ve Hazırlama

Bu döküman, bir Makine Öğrenimi (ML) projesinin uçtan uca yaşam döngüsünü, karşılaşılan temel veri türlerini ve model başarısı için hayati önem taşıyan özellik hazırlama (Feature Preparation) adımlarını özetler.

---

## I. 🔄 ML Yaşam Döngüsüne Genel Bakış (ML Lifecycle)

Bir ML çözümü geliştirmek, problem tanımından (Problem Definition) başlayıp, sürekli takiple biten yedi aşamalı döngüsel bir süreçtir.

| Aşama No. | Aşama Adı | Görev Tanımı ve Amacı | Örnek Uygulama (Müşteri Kaybı Çözümünde) |
| :---: | :--- | :--- | :--- |
| **1** 🎯 | **Problem Definition** (Problem Tanımı) | Net bir soru belirleyerek projenin amacını tanımlar. Başarılı bir ML projesi için gerekli ilk adımdır. | **Soru:** "Müşteriler bizi neden terk ediyor ve tam olarak neyi tahmin etmek istiyoruz?". |
| **2** 🎁 | **Data Collection** (Veri Toplama) | Tanımlanan problem için gerekli olan ilgili verileri (çeşitli kaynaklardan) toplar. | İşlem günlükleri, müşteri profilleri ve destek hattı verilerini toplamak. |
| **3** 🧹 | **Data Preparation** (Veri Hazırlama) | Ham veriler dağınıktır; bu aşamada veriler temizlenir, organize edilir ve ML modelinin kullanabileceği formata dönüştürülür. | Eksik değerleri temizlemek, çift kayıtları (duplicate) kaldırmak ve "müşteri bekleme süresi" gibi yeni özellikler oluşturmak. |
| **4** 🔍 | **Data Exploration** (Veri Keşfi) | Model kurulmadan önce veriyi anlamak, kalıpları, eğilimleri ve aykırı değerleri (outlier) görseleştirmelerle ve istatistiklerle keşfetmek. | Müşterilerin ayda 3 kereden fazla destek birimiyle iletişime geçtiğinde ayrılma olasılığının daha yüksek olduğunu keşfetmek. |
| **5** 🧠 | **Model Building** (Model Kurulumu) | Uygun algoritma seçilir ve tarihsel (eğitim) verileri kullanılarak model eğitilir. | Lojistik Regresyon gibi bir model seçmek ve bu modeli tarihsel kayıp (churn) verileriyle eğitmek. |
| **6** ⚖️ | **Model Evaluation** (Model Değerlendirme) | Modelin **görülmemiş test verileri** üzerindeki performansını kontrol eder. Performans düşüklüğünde, model hiper parametreleri ayarlanır veya başka algoritmalar denenir. | Doğruluk (**accuracy**) ve geri çağırma (**recall**) metrikleriyle performansı ölçmek, ardından hiper parametreleri ayarlayarak sonuçları iyileştirmek. |
| **7** 🚀 | **Model Deployment & Monitoring** (Model Dağıtımı & Takibi) | Model gerçek zamanlı tahminler yapabilmesi için üretim ortamına alınır (deploy edilir). İş bitmemiştir; modelin zamanla değişen veri kalıplarına uyum sağlaması için izlenmesi ve gerektiğinde **yeniden eğitilmesi** gerekir. | Modeli üretime almak, böylece satış ekibinin anlık kayıp riski puanlarını görmesini sağlamak ve kalıp kalıpları değiştikçe modeli izleyip yeniden eğitmek. |

---

## II. 💡 Makine Öğreniminde Temel Veri Türleri (Key Data Types)

Veri türünü bilmek, **ön işleme (preprocessing method)** ve hangi algoritmaların uygulanacağını belirten ilk adımdır.

| Ana Veri Tipi | Alt Tip | Açıklama ve Önemi (Why it matters?) | Örnek Senaryo |
| :--- | :--- | :--- | :--- |
| **Sayısal** (Numeric / Quantitative) | **Sürekli** (Continuous) | Bir aralıkta herhangi bir gerçek değeri alabilir. Örn: ağırlık, boy, sıcaklık. Sayılar çok farklı ölçeklerde olabilir; modelin büyük sayılara daha fazla önem vermemesi için ayarlanmalıdır. | Günlük Apple hisse senedi kapanış fiyatı. |
| | **Kesikli** (Discrete) | Sayılabilir miktarları temsil eder ve yalnızca tam sayı değerleri alır. Örn: satılan ürün sayısı, mağazadaki müşteri sayısı. | Satılan bilet sayısı. |
| **Kategorik** (Categorical / Qualitative) | **Nominal** (Nominal) | Sıralaması olmayan kategoriler. Örn: cinsiyet, renkler. | Kırmızı, Mavi, Yeşil. |
| | **Sırasal** (Ordinal) | Anlamlı bir rütbe veya sıraya sahip kategoriler. Örn: eğitim seviyeleri, derecelendirmeler ("poor, fair, good, excellent"). | Müşteri memnuniyet anketindeki puanlar. |
| **Metin Verisi** (Text Data) | (Unstructured) | İncelemeler, tweet'ler veya e-postalar gibi yapılandırılmamış kelimeler ve cümleler. | Bilgisayarlar metni sadece karakter dizileri olarak görür. Öğrenmesi için metni sayılarla (örn: kelime sayıları) temsil etmemiz gerekir. | IMDb'den film incelemelerinin listesi. |
| **Zaman Serisi** (Time Series Data) | (Continuous) | Düzenli aralıklarla (günlük, aylık vb.) toplanan veriler. Örn: hisse senedi fiyatları, sensör okumaları. | Sıra önemlidir. Dünün değeri bugünü etkiler ve iyi tahminler yapmak için trendlerin yakalanması gerekir. | Günlük Apple hisse senedi kapanış fiyatı. |
| **İkili** (Boolean / Binary Data) | (Binary) | İki olası değer: 0/1 veya True/False. Örn: spam/spam değil, geçme/kalma. | İkili özellikler zaten sayısal olmasına rağmen, bir sınıf nadir ise (Örn: %2 e-posta spam), modelin her zaman "spam değil" tahmin ederek hile yapmaması için dengeleme gerektirir. | Müşterinin bir reklama tıklayıp tıklamadığını gösteren sütun ("1" tıklandı, "0" tıklanmadı). |
| **Görüntü Verisi** (Image Data) 🖼️ | (Pixels) | Piksellerden oluşur (gri tonlamalı veya RGB). | Tek bir görüntü **milyonlarca sayı** (piksel) içerebilir. Modelin aşırı yüklenmemesi için görüntülerin özel tekniklerle işlenmesi gerekir. | Bir hastanın beyninin MRI taraması. |

---

## III. 🛠️ Özellik Hazırlamada Temel Adımlar (Feature Preparation Steps)

Özellik Hazırlama (**Feature Preparation**) ham veriyi temiz, yapılandırılmış ve tutarlı bir forma dönüştüren kritik süreçtir.

| Adım No. | Aşama Adı (İngilizce Terim) | Amaç ve Açıklama | Somut Örnekler |
| :---: | :--- | :--- | :--- |
| **1** ❓ | **Handling Missing Data** (Eksik Veri Yönetimi) | Veri setinde hiç veri girişi olmayan boş hücreleri (NaN) ele alma. Veri kaybını en aza indirerek veri setinin bütünlüğünü korumak. | **Sayısal:** Yaş (**Age**) verisindeki boşlukları, ortalama (**mean**) veya medyan (**median**) ile doldurmak (**Imputation**).<br>**Kategorik:** Eksik değeri "**Bilinmiyor**" (**Unknown**) adında yeni bir kategori olarak işaretlemek. |
| **2** ❗ | **Handling Outliers** (Aykırı Değer Yönetimi) | Veri setinin geri kalanından önemli ölçüde farklı olan aşırı değerleri tespit etmek ve düzeltmek. Modelin bu uç değerlerden yanlış öğrenmesini engellemek. | Gelir verisinde 1.000.000.000 USD gibi bir değerin tespiti. Bu değeri kaldırabilir veya kabul edilebilir bir üst sınırla (**capping**) değiştirebiliriz. |
| **3** 🏷️ | **Handling Categorical Data** (Kategorik Veri Yönetimi) | Metin tabanlı kategorik özellikleri (Örn: şehir adları, renkler) ML algoritmalarının anlayabileceği sayısal formata çevirme. | **Nominal:** "Kırmızı", "Mavi", "Yeşil" gibi sırasız renkler için **One-Hot Encoding** kullanmak.<br>**Ordinal:** "Kötü", "Orta", "İyi" gibi sıralı derecelendirmeler için **Label Encoding** kullanmak (1, 2, 3 gibi). |
| **4** ⚖️ | **Feature Scaling** (Özellik Ölçeklendirme) | Sayısal özelliklerin değer aralıklarını ortak bir standarda getirmek. Modelin, büyük değer aralığına sahip özelliklere haksız yere daha fazla önem vermesini önler. | **Normalizasyon (Normalization):** Veriyi 0 ile 1 arasına ölçeklendirme.<br>**Standartizasyon (Standardization):** Veriyi ortalaması 0 ve standart sapması 1 olacak şekilde dönüştürme. |
| **5** ✨ | **Feature Creation and Transformation** (Özellik Oluşturma ve Dönüştürme) | Mevcut özelliklerden yeni ve daha bilgilendirici özellikler türetme veya mevcut özellikleri dönüştürme. Modelin öğrenmesine yeni bakış açıları katmak. | Müşterinin doğum tarihinden "**Müşteri Yaşı**" veya "**Müşteri Olma Süresi**" gibi yeni bir özellik türetme. İki sütunu çarparak yeni bir etkileşim terimi (**interaction term**) oluşturma. |
| **6** 🎯 | **Feature Selection** (Özellik Seçimi) | Tahmin hedefiyle en ilgili olan özelliklerin alt kümesini seçme. Alakasız veya gereksiz özellikleri kaldırarak modelin karmaşıklığını ve eğitim süresini azaltmak. | Bir ev fiyatı tahmini modelinde, adresin kapı numarasını veya rengini kaldırıp, sadece metrekare ve oda sayısı gibi daha alakalı özelliklere odaklanmak. |

## 🛠️ Feature Preparation Steps and Techniques

| Step No. | Phase (English Term) | Primary Goal | Key Methods and Comparison |
| :---: | :--- | :--- | :--- |
| **1** ❓ | **Handling Missing Data** (Eksik Veri Yönetimi) | To handle missing values (NaN/Null) to maintain dataset integrity and ensure the model can function. | **1. Deletion (Kaldırma):** Used only when very few rows/columns are missing. **2. Imputation (Doldurma):** Filling missing data with statistical values:<br>— **Numeric:** Filling with Mean, Median, or Group Mean.<br>— **Categorical:** Filling with the Mode or a new category like "Unknown." |
| **2** ❗ | **Handling Outliers** (Aykırı Değer Yönetimi) | To prevent extreme values (outliers) from distorting model learning and predictions. | **1. Detection:** Using Boxplots, Z-Score, or IQR method.<br>**2. Treatment:**<br>— **Removal (Kaldırma):** Removing the observations (risk of data loss).<br>— **Capping/Winsorization (Kısıtlama):** Setting the value at a specific upper or lower threshold.<br>— **Transformation (Dönüştürme):** Using Log Transform to normalize the distribution. |
| **3** 🏷️ | **Handling Categorical Data** (Kategorik Veri Yönetimi) | To convert text-based categorical features (Nominal/Ordinal) into a numerical format that ML algorithms can process. | **1. One-Hot Encoding:** Ideal for **Nominal** (unordered) data. Creates a new binary column for each category.<br>**2. Label Encoding:** Ideal for **Ordinal** (ranked) data. Assigns numerical ranks to categories (e.g., Poor=1, Good=2).<br>**3. Target Encoding:** Effective for high-cardinality (many unique values) categorical data. |
| **4** ⚖️ | **Feature Scaling** (Özellik Ölçeklendirme) | To bring the value ranges of numerical features to a common standard, preventing the model from unfairly prioritizing features with larger ranges. | **1. Normalization (Min-Max Scaling):** Scales values between **0 and 1**. Sensitive to outliers.<br>**2. Standardization (Z-Score Scaling):** Transforms data to a distribution with a mean of **0** and a standard deviation of **1**. More robust to outliers. |
| **5** ✨ | **Feature Creation and Transformation** (Özellik Oluşturma ve Dönüştürme) | To derive new, more informative features or transform existing ones to improve model learning. | **1. Feature Extraction:** Deriving new information (e.g., **Age** or **Customer Tenure** from Date of Birth).<br>**2. Binning:** Grouping a continuous feature into categorical bins (e.g., Income into Low, Medium, High groups).<br>**3. Interaction Features:** Creating a new feature from the combination/multiplication of two different features. |
| **6** 🎯 | **Feature Selection** (Özellik Seçimi) | To select a subset of features most relevant to the prediction goal, reducing model complexity and training time. | **1. Filter Methods:** Selecting features based on statistical tests (e.g., correlation, Chi-square).<br>**2. Wrapper Methods:** Searching for the best feature combination using the model itself (e.g., RFE).<br>**3. Embedded Methods:** The model determines feature importance during training (e.g., Lasso). |









----
# Proposal for Improving Practice Sessions and Portfolio Development

This table outlines suggested changes to the practice session structure, focusing on active learning, portfolio readiness, and clear learning outcomes.

| Category | Proposal  | Proposal (Added Strategy) | Benefit & Outcome |
| :--- | :--- | :--- | :--- |
| **Structured Output & Accountability** | **1. Weekly Mini-Projects/Assignments:** Assign clear, weekly mini-projects or challenging questions related to the covered topic. | **2. Coursera/Online Project Integration:** Select a project from platforms like Coursera related to the current topic and complete it collaboratively with the instructor's guidance. | **Clarity and Portfolio Building:** Provides concrete, measurable tasks. Ensures practical application of theory, leading to ready-to-use **portfolio pieces**. Converts abstract knowledge into tangible results. |
| **Interactive Skill Development** | *(Implicit in Group Work)* | **3. Live Code Review and Debugging:** Students share their code live for peer/instructor review, focusing on finding errors and optimization. | **Debugging Proficiency:** Sharpens critical debugging and **code optimization** skills. Exposes students to diverse problem-solving approaches beyond a single "correct" answer. |
| **Problem Solving & Analysis** | *(Implicit in Group Work)* | **4. Reverse Engineering Exercises:** Analyze, explain, and refactor a complex or inefficient piece of existing code provided by the instructor. | **System Analysis:** Develops the essential real-world skill of analyzing and **improving existing systems (refactoring)**. Encourages focused deconstruction of complex topics. |
| **Active Learning Format** | *(Implicit in Group Work)* | **5. Workshop Format (Tech Kitchens):** Convert practice sessions into goal-oriented workshops where students work on a specific challenge. The instructor acts solely as a **mentor**. | **Engagement & Mentorship:** Shifts the student from passive listener to active practitioner. Maximizes the instructor's value by focusing them on providing targeted help where groups are genuinely stuck. |
| **Knowledge Consolidation** | *(Implicit in Asking Questions)* | **6. Student-Led Topic Presentation Rotation:** Different groups present the most challenging part of the previous week's lecture to the class. | **Mastery Through Teaching:** Utilizes the **Feynman Technique**; teaching a subject is the best way to confirm true mastery. Fosters a clearer understanding among peers. |
