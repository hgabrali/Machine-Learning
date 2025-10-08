# Features Preparation


* Features are the inputs you give to a machine learning model so it can make predictions. They’re the measurable properties or characteristics that describe each data point.

  <img width="563" height="329" alt="image" src="https://github.com/user-attachments/assets/07bb778d-d2af-42d1-9119-bd9be9d612ef" />

## 🛠️ Özellik Hazırlamada Temel Adımlar (Feature Preparation Steps)

Özellik Hazırlama (**Feature Preparation**) ham veriyi temiz, yapılandırılmış ve tutarlı bir forma dönüştüren kritik süreçtir.

| Adım No. | Aşama Adı (İngilizce Terim) | Amaç ve Açıklama | Somut Örnekler |
| :---: | :--- | :--- | :--- |
| **1** ❓ | **Handling Missing Data** (Eksik Veri Yönetimi) | Veri setinde hiç veri girişi olmayan boş hücreleri (NaN) ele alma. Veri kaybını en aza indirerek veri setinin bütünlüğünü korumak. | **Sayısal:** Yaş (**Age**) verisindeki boşlukları, ortalama (**mean**) veya medyan (**median**) ile doldurmak (**Imputation**).<br>**Kategorik:** Eksik değeri "**Bilinmiyor**" (**Unknown**) adında yeni bir kategori olarak işaretlemek. |
| **2** ❗ | **Handling Outliers** (Aykırı Değer Yönetimi) | Veri setinin geri kalanından önemli ölçüde farklı olan aşırı değerleri tespit etmek ve düzeltmek. Modelin bu uç değerlerden yanlış öğrenmesini engellemek. | Gelir verisinde 1.000.000.000 USD gibi bir değerin tespiti. Bu değeri kaldırabilir veya kabul edilebilir bir üst sınırla (**capping**) değiştirebiliriz. |
| **3** 🏷️ | **Handling Categorical Data** (Kategorik Veri Yönetimi) | Metin tabanlı kategorik özellikleri (Örn: şehir adları, renkler) ML algoritmalarının anlayabileceği sayısal formata çevirme. | **Nominal:** "Kırmızı", "Mavi", "Yeşil" gibi sırasız renkler için **One-Hot Encoding** kullanmak.<br>**Ordinal:** "Kötü", "Orta", "İyi" gibi sıralı derecelendirmeler için **Label Encoding** kullanmak (1, 2, 3 gibi). |
| **4** ⚖️ | **Feature Scaling** (Özellik Ölçeklendirme) | Sayısal özelliklerin değer aralıklarını ortak bir standarda getirmek. Modelin, büyük değer aralığına sahip özelliklere haksız yere daha fazla önem vermesini önler. | **Normalizasyon (Normalization):** Veriyi 0 ile 1 arasına ölçeklendirme.<br>**Standartizasyon (Standardization):** Veriyi ortalaması 0 ve standart sapması 1 olacak şekilde dönüştürme. |
| **5** ✨ | **Feature Creation and Transformation** (Özellik Oluşturma ve Dönüştürme) | Mevcut özelliklerden yeni ve daha bilgilendirici özellikler türetme veya mevcut özellikleri dönüştürme. Modelin öğrenmesine yeni bakış açıları katmak. | Müşterinin doğum tarihinden "**Müşteri Yaşı**" veya "**Müşteri Olma Süresi**" gibi yeni bir özellik türetme. İki sütunu çarparak yeni bir etkileşim terimi (**interaction term**) oluşturma. |
| **6** 🎯 | **Feature Selection** (Özellik Seçimi) | Tahmin hedefiyle en ilgili olan özelliklerin alt kümesini seçme. Alakasız veya gereksiz özellikleri kaldırarak modelin karmaşıklığını ve eğitim süresini azaltmak. | Bir ev fiyatı tahmini modelinde, adresin kapı numarasını veya rengini kaldırıp, sadece metrekare ve oda sayısı gibi daha alakalı özelliklere odaklanmak. |
5.  **Veriyi Ayırma (Splitting data) 🪓:** Veriyi eğitim (training) ve test (test) setlerine bölme.

> **Unutmayın:** Başarılı bir ML projesinin temeli, daima **temizlenmiş ve doğru şekilde hazırlanmış** veriye dayanır.




<img width="827" height="513" alt="image" src="https://github.com/user-attachments/assets/0214c465-8578-49a7-802a-4ffe3d6ef59d" />

<img width="801" height="477" alt="image" src="https://github.com/user-attachments/assets/9601f514-bb33-4be4-9bfd-bc982d76f904" />

<img width="804" height="560" alt="image" src="https://github.com/user-attachments/assets/79a34d9e-f0de-4afe-b28c-12a147ccd74d" />


<img width="920" height="459" alt="image" src="https://github.com/user-attachments/assets/884a702f-b29e-4350-bc86-ef002a69efc1" />


# 🛠️ Data Preparation Techniques for Machine Learning

Data preparation is the crucial process of transforming raw data into a form more suitable for modeling. The required steps depend on the specific data and the algorithms to be used.

---

## The 5 Core Data Preparation Tasks

These tasks form the framework for preparing structured (tabular) data in a predictive modeling project:

### 1. Data Cleaning 🧹
* **Definition:** Identifying and correcting mistakes or errors in "messy" data.
* **Goal:** Ensure data quality and reliability.
* **Key Operations:**
    * Identifying and addressing **outliers** (using statistical methods).
    * Removing **duplicate rows** of data.
    * Identifying columns with zero variance (same value) and removing them.
    * Marking empty values as missing and **imputing** (filling) missing values using statistics (mean, median) or a learned model.

### 2. Feature Selection 🎯
* **Definition:** Selecting a subset of input features that are most relevant to the target variable.
* **Goal:** Improve model performance and favor the simplest possible model by removing **irrelevant and redundant variables**.
* **Technique Groups:**
    * **Filter Methods:** Scoring input features (e.g., using correlation statistics) and selecting the top subset.
    * **Wrapper Methods:** Explicitly choosing features that result in the best-performing model (e.g., RFE).
    * **Intrinsic Methods:** Models that automatically select features during the fitting process.

### 3. Data Transforms 🔄
* **Definition:** Changing the type or distribution of variables to meet algorithm requirements.
* **Key Transforms:**
    * **Encoding Categorical Data:** Converting non-numeric labels into a numeric form:
        * *One-Hot Transform:* For nominal (unordered) variables.
        * *Ordinal Transform:* For ordinal (ranked) variables.
    * **Scaling Numeric Data:** Adjusting the range of real-valued variables:
        * **Normalization:** Scaling a variable to a range between 0 and 1.
        * **Standardization:** Shifting data to a Standard Gaussian (mean of zero, std dev of one).
    * **Power Transform / Quantile Transform:** Changing the probability distribution of numerical variables (e.g., making the distribution more Gaussian).

### 4. Feature Engineering ✨
* **Definition:** Creating **new input variables** from the available data. This often requires deep **domain expertise**.
* **Goal:** Provide the model with a more straightforward perspective on the input data and add broader context.
* **Examples:**
    * Adding a **boolean flag** for a specific state (e.g., *IsWeekend*).
    * Deriving **summary statistics** (e.g., adding a global mean).
    * Decomposing complex variables (e.g., splitting a **date-time** into separate Day, Month, Year variables).

### 5. Dimensionality Reduction 📉
* **Definition:** Creating a projection of the data into a **lower-dimensional space** while preserving the most important properties.
* **Goal:** Address the **"curse of dimensionality"** (when too many input variables lead to a sparse and unrepresentative sampling of the space).
* **Key Techniques:**
    * **Principal Component Analysis (PCA)**
    * **Singular Value Decomposition (SVD)**

> **Note:** Unlike Feature Selection, variables created by dimensionality reduction are not directly related to the original inputs, making the results difficult to interpret.

 [Source: ML Mastery ,Jason Brownlee PhD](https://machinelearningmastery.com/data-preparation-techniques-for-machine-learning/)
