

<img width="1504" height="780" alt="image" src="https://github.com/user-attachments/assets/df684f88-a47d-44ec-81db-f22c352eac2f" />

* [Picture](https://www.labellerr.com/blog/supervised-vs-unsupervised-learning-whats-the-difference/)

# Supervised vs. Unsupervised Learning: A Technical Comparison ⚖️

This table details the key differences in data, goal, algorithm, and evaluation between Supervised and Unsupervised Learning.

| Feature | Supervised Learning 🎯 (Denetimli Öğrenme) | Unsupervised Learning 🔍 (Denetimsiz Öğrenme) |
| :--- | :--- | :--- |
| **Data Type** 💾 | **Labeled Data** ($\mathbf{X}$, $\mathbf{y}$). Each input ($\mathbf{X}$) is paired with a corresponding output label ($\mathbf{y}$). | **Unlabeled Data** ($\mathbf{X}$). Only the input features are available; there is no target label ($\mathbf{y}$). |
| **Primary Goal** ➡️ | **Prediction/Mapping.** The model learns a function to map inputs to known outputs ($f(\mathbf{X}) \rightarrow \mathbf{y}$). | **Discovery/Structure.** The model aims to infer the underlying patterns, hidden structures, or natural groupings within the data. |
| **Typical Problems** 🧩 | **Classification** (predicting discrete labels) and **Regression** (predicting continuous values). | **Clustering** (grouping similar data points), **Dimensionality Reduction** (data compression), and **Association** (finding relationships). |
| **Model Training Process** ⚙️ | Training is done using an **"external supervisor"** (the $\mathbf{y}$ labels) to guide the learning process and correct errors during iterative training. | Training is **autonomous** and exploratory. The model seeks similarity and distribution patterns entirely on its own. |
| **Common Algorithms** 🧮 | Linear Regression, Logistic Regression, Support Vector Machines (SVM), k-Nearest Neighbors (k-NN), Decision Trees, Random Forest, Neural Networks. | K-Means Clustering, Hierarchical Clustering, DBSCAN, Principal Component Analysis (PCA), t-SNE, Autoencoders, Apriori Algorithm. |
| **Evaluation Metrics** 💯 | Based on a known **Ground Truth**. Metrics include Accuracy, Precision, Recall, F1-Score (for Classification), and RMSE, MAE, $R^2$ (for Regression). | Lacks a Ground Truth for external validation. Metrics focus on **internal consistency** (e.g., Silhouette Score, Davies–Bouldin Index) or application-specific measures. |
| **Computational Requirement** ⏱️ | Can be computationally **intensive** during the training phase due to optimizing complex loss functions against known labels. | Can be computationally **intensive** when dealing with high-dimensional data or complex clustering (e.g., large-scale distance calculations). |
| **Real-World Examples** 🌍 | Spam detection, image recognition (labeling an object), house price prediction, sentiment analysis. | Customer segmentation, anomaly/fraud detection, data compression for visualization, market basket analysis. |

# 🧠 Unsupervised Learning: Discovering Structure in Unlabeled Data

## 1. Introduction: The Need for Autonomy

Traditional **Supervised Learning** relies on datasets where each instance contains both **features ($\mathbf{X}$)** (input data, e.g., a passenger's age, class, sex) and a corresponding **target ($\mathbf{y}$)** (the ground-truth label, e.g., 'survived' or 'not survived').

However, in many real-world scenarios, the target labels are unavailable, impractical to obtain, or simply nonexistent. This presents the fundamental challenge: **how can a computational system derive meaningful insights, patterns, or intrinsic structures from raw, unlabeled data entirely on its own?**

This autonomy is the domain of **Unsupervised Learning**.

***

## 2. Technical Definition and Core Objective

**Unsupervised Learning (UL)** is a category of Machine Learning algorithms designed to infer a function to describe hidden structure from **unlabeled data** ($\mathbf{X}$) — data that has no associated output values ($\mathbf{y}$).

* **Input:** Only the **feature matrix ($\mathbf{X}$)**.
* **Goal:** To model the underlying distribution, density, or latent variables of the data to facilitate:
    1.  **Pattern Discovery:** Identifying recurring relationships or associations.
    2.  **Dimensionality Reduction:** Compressing the feature space while retaining essential information.
    3.  **Clustering:** Grouping similar data points together.

### Core Analogy (Dimensionality/Similarity)

Consider a large, unsorted dataset of fruits. A UL algorithm doesn't know the human-defined labels ('apple', 'watermelon'). Instead, it analyzes the feature space (e.g., color, size, shape, weight) and identifies **natural separations** where intra-group variance is minimized and inter-group variance is maximized. This process defines the intrinsic structure of the data.

***

## 3. The Rationale: Why Unlabeled Data Persists

The prevalence of Unsupervised Learning is driven by the significant practical constraints associated with data labeling (or **annotation**):

| Constraint | Description | Example |
| :--- | :--- | :--- |
| **Cost & Scale** | Manual labeling of vast datasets (Big Data) is prohibitively expensive and time-consuming, requiring extensive human effort. | Annotating millions of high-resolution images or hours of streaming audio. |
| **Expert Dependency** | Specialized domains require highly skilled subject-matter experts for accurate annotation, which are scarce resources. | Labeling rare astronomical events or complex genetic sequences. |
| **Epistemological Gap** | In exploratory data analysis (e.g., frontier science), the ground truth is **unknown**. If the underlying phenomenon is not yet understood, no meaningful label can be assigned. | Identifying novel cell types or emergent patterns in high-dimensional genomic data. |

UL algorithms bypass this dependency, providing the initial step to structure raw data, which can then potentially be used in a semi-supervised or domain-specific application.

***

## 4. Key Use Cases and Applications

Unsupervised learning excels in tasks aimed at simplification, segmentation, and novelty detection:

### A. Clustering (Segmentation)

The process of grouping a set of data objects into subsets (clusters) so that objects within the same cluster are more similar to each other than to those in other clusters.

* **Application:** **Customer Segmentation** (e.g., *k*-Means, DBSCAN) where a business wants to identify inherent market segments (e.g., "High-Value Spenders," "Occasional Browsers") without pre-defining the categories.

### B. Dimensionality Reduction

Reducing the number of random variables under consideration by obtaining a set of principal variables. This combats the **curse of dimensionality** and aids visualization.

* **Application:** **Feature Extraction** (e.g., Principal Component Analysis **PCA**, t-distributed Stochastic Neighbor Embedding **t-SNE**). Used to simplify complex genomic or image datasets for faster processing and better model interpretability.

### C. Association Rule Mining

Discovering interesting relationships between variables in large databases.

* **Application:** **Market Basket Analysis** (e.g., Apriori algorithm). Identifying products frequently purchased together (e.g., *if customer buys A and B, they are likely to buy C*).

# Real-World Applications of Unsupervised Learning (Clustering, Association, and Reduction)

| Application Area (Emoji) | Core Unsupervised Technique | Detailed Real-World Example / Benefit |
| :--- | :--- | :--- |
| **Marketing & E-commerce** 🛍️ | **Clustering** (K-Means, Hierarchical) | **Customer Segmentation:** Automatically groups customers into distinct segments (e.g., "High Spenders," "Discount Seekers") based on their purchasing history, demographics, and web behavior. *Benefit:* Allows businesses to tailor personalized marketing campaigns and product offerings, significantly boosting conversion rates and customer loyalty. |
| **Finance & Security** 💰 | **Anomaly Detection** (Isolation Forest, DBSCAN) | **Fraud and Intrusion Detection:** Identifies rare, unusual data points or patterns that deviate significantly from the established 'normal' behavior within financial transactions or network traffic logs. *Benefit:* Crucial for flagging potential credit card fraud, money laundering activities, or unauthorized network access in real-time. |
| **Media & Content** 📺 | **Clustering** & **Association** | **Recommendation Systems:** Groups users with similar consumption habits (e.g., movie genres watched, products bought) to provide highly relevant suggestions. **Matrix Factorization** (a form of dimensionality reduction) is often used to discover latent features. *Example:* Netflix suggesting new titles based on similar viewers' preferences. |
| **Retail & Inventory** 🛒 | **Association Rule Mining** (Apriori) | **Market Basket Analysis (MBA):** Discovers strong relationships and "if-then" rules between items frequently purchased together in large transaction datasets. *Benefit:* Optimizes store layout, product bundling, and cross-selling strategies (e.g., placing bread next to butter). |
| **Data Preprocessing** 📊 | **Dimensionality Reduction** (PCA, t-SNE) | **Data Simplification & Visualization:** Reduces the number of variables (features) in high-dimensional datasets while preserving their most critical variance. *Benefit:* Eliminates redundant data, speeds up subsequent supervised learning models, and makes complex data understandable for human visualization (e.g., plotting high-dimensional clusters on a 2D map). |
| **Natural Language Processing** 📰 | **Topic Modeling** (LDA) & **Clustering** | **Document Organization and Thematic Grouping:** Automatically categorizes massive, unstructured text collections (e.g., news articles, research papers) into overarching themes or topics without needing predefined tags. *Example:* Google News aggregating related stories from thousands of sources under a single category. |
| **Healthcare & Genomics** 🧬 | **Clustering** & **Dimensionality Reduction** | **Patient Subgroup Identification:** Analyzes complex genetic, proteomic, or clinical data to identify natural, previously unknown subgroups of patients with similar disease characteristics. *Benefit:* Aids in discovering new disease subtypes and moving toward personalized medicine and targeted drug development. |


# Why Unsupervised Learning Matters (Denetimsiz Öğrenme Neden Önemlidir) 🧠

## Why Do We Need Unsupervised Learning?

Unsupervised learning is essential because it allows us to make sense of **data without labels**. It provides a powerful set of tools when human intervention for labeling is impractical or impossible. Here are some key reasons why it matters:

### 1. Exploratory Analysis ✨

When we don’t know much about a dataset, unsupervised learning helps us discover hidden **relationships, patterns, or groups**. It acts as a guide in the early stages of data analysis.

* **Example:** Finding natural **clusters** of customers in sales data, revealing market segments you didn't know existed.

### 2. Handling Massive, Unlabeled Data 💾

Most of the data generated today—social media posts, healthcare records, financial transactions—is **unlabeled** and far too large to tag manually.

* Unsupervised algorithms can organize and simplify these overwhelming datasets through techniques like **Dimensionality Reduction** (e.g., PCA) and **Clustering**, making them manageable for further analysis.

### 3. Unlocking Hidden Insights 🔓

Sometimes the most valuable findings are the ones we didn’t expect. Unsupervised learning can uncover **anomalies, trends, and structures** we didn’t even know to look for, leading to genuine discovery (e.g., detecting novel fraud patterns).

---

## Challenges in Unsupervised Learning ⚠️

While powerful, unsupervised learning isn’t perfect. Because the data has no ground truth, it presents unique challenges:

| Challenge (Emoji) | Explanation |
| :--- | :--- |
| **Interpreting the Results** 🤔 | Without labels, it’s up to humans (domain experts) to decide whether the discovered clusters, components, or patterns are actually meaningful and actionable in a business or scientific context. |
| **No Clear Evaluation Metric** 📉 | Unlike supervised learning (where we use accuracy or error rates), there’s no single **ground truth** to compare against. Evaluation often relies on subjective measures or internal metrics like silhouette scores. |
| **Ambiguity** 🤷 | Different algorithms (or even different runs of the same stochastic algorithm) may produce slightly different groupings. The output is often less deterministic than in supervised models. |
| **Computational Complexity** ⚙️ | Especially in high-dimensional space, algorithms like Hierarchical Clustering can be very **resource-intensive** to process, making them slow for massive datasets. |

---

## Summary 🎉

* **Unsupervised learning** = **no labels** ($y$ is missing).
* It’s about finding **patterns, groups, or structures** in data.
* Very useful in the real world when labels are **unavailable or too costly** to acquire.



# Key Differences: Supervised vs. Unsupervised Learning

In the previous lesson, we introduced the concept of **Unsupervised Learning** ($U$) where models operate on **unlabeled data** ($X$) to extract inherent patterns. Now, let's establish a technical comparison with **Supervised Learning** ($S$), which you are already familiar with.

---

## 1. Data Type: Labeled vs. Unlabeled 🏷️

The most fundamental distinction between $S$ and $U$ learning lies in the nature of the training data:

| Feature | Supervised Learning ($S$) | Unsupervised Learning ($U$) |
| :--- | :--- | :--- |
| **Data Structure** | Requires **labeled data** ($X, y$), where each input vector $X_i$ has a corresponding output label or value $y_i$. | Operates solely on **unlabeled input data** ($X$), where the target variable $y$ is unknown. |
| **Model Function** | Learns a mapping function $f: X \rightarrow Y$. | Aims to model the underlying structure or distribution of the data $P(X)$. |

Let's visualize the raw data difference:

| Supervised Data (Classification) | Unsupervised Data (Clustering) |
| :---: | :---: |
| Data points are distinctly color-coded, representing known classes (e.g., Cat vs. Dog).  | All data points are uniformly colored, challenging the model to discover separation.  |

---

## 2. Goal: Prediction vs. Discovery 🎯

The end objective dictates the choice of the learning paradigm:

| Goal Type | Supervised Learning ($S$) | Unsupervised Learning ($U$) |
| :--- | :--- | :--- |
| **Primary Aim** | **Prediction** or **Estimation**. The goal is to accurately predict known outcomes ($y$) for new, unseen inputs ($X$). | **Discovery** or **Inference**. The goal is to find hidden structures, groupings, or representations within the data. |
| **Metric Focus** | Minimizing prediction error (e.g., $L_2$ loss for Regression). | Optimizing a measure of data structure quality (e.g., cluster compactness/separation). |

Here's an illustrative example contrasting the goals:

| Supervised Goal: House Price Prediction (Regression) | Unsupervised Goal: Customer Clustering |
| :---: | :---: |
| A red line shows the learned linear relationship between house size ($X$) and predicted price ($y$) based on labeled data.  | Data points are grouped into distinct segments (colors), reflecting underlying customer patterns without prior labels.  |

---

## 3. Problem Types and Applications 🛠️

The absence or presence of a target variable $y$ inherently limits the types of problems each approach can address:

| Problem Type | Supervised Learning ($S$) | Unsupervised Learning ($U$) |
| :--- | :--- | :--- |
| **Common Tasks** | **Classification** (predicting discrete labels) and **Regression** (predicting continuous values). | **Clustering** (grouping similar data points) and **Dimensionality Reduction** (simplifying feature space). |
| **Examples** | Spam Detection, Image Recognition, Stock Price Forecasting. | Customer Segmentation, Anomaly Detection, Feature Extraction. |

A visual contrast of the problem output:

| Supervised Task: Classification | Unsupervised Task: Dimensionality Reduction & Clustering |
| :---: | :---: |
| A decision boundary separates data points into two distinct, predefined categories (e.g., Spam vs. Non-Spam).  | Data is reduced (e.g., using PCA) and grouped into natural, emergent clusters.  |

---

## 4. Model Complexity and Evaluation 📈

Model architecture and validation methods also differ significantly:

### Model Complexity and Architecture

* **Supervised Learning:** Often employs more **complex models** (e.g., Deep Neural Networks, Random Forests, Support Vector Machines) designed for precise, high-dimensional **mapping** and robust generalization.
* **Unsupervised Learning:** Typically relies on **simpler algorithms** (e.g., K-Means, PCA, Autoencoders) focused on finding an optimal internal **representation** or latent structure of the data.

### Evaluation Metrics

* **Supervised Learning:** Performance is evaluated against the **ground truth ($y$)** using objective external metrics: **Accuracy, Precision, Recall, F1 Score** (for classification), or **RMSE** (Root Mean Square Error, for regression).
* **Unsupervised Learning:** Due to the lack of $y$, evaluation relies on **intrinsic metrics** that measure the quality of the discovered structure: **Silhouette Score** (cluster separation and compactness), **Davies–Bouldin Index**, or Reconstruction Error (for dimensionality reduction). Interpretation often requires **human domain expertise** to validate the utility of the discovered patterns.




---
# TÜRKCE:

# Denetimsiz Öğrenme Nedir? (Unsupervised Learning)

## 1. Giriş ve Bağlam

Şu ana kadar, verinin hem **Özniteliklere** ($X$) hem de **Hedefe** ($y$) sahip olduğu **denetimli öğrenme** (supervised learning) konseptiyle çalıştık.

* **Öznitelikler ($X$)**: Giriş verisi (örn. Titanic veri setindeki yolcunun yaşı, sınıfı, cinsiyeti).
* **Hedef ($y$)**: Tahmin etmek istediğimiz çıktı (örn. hayatta kalıp kalmadığı).

Peki ya elimizde etiketler yoksa? Ya sadece ham verimiz varsa ve bilgisayarın **kendi başına** örüntüleri, yapıları veya grupları keşfetmesini istiyorsak?

İşte tam bu noktada **Denetimsiz Öğrenme** devreye girer.

***

## 2. Teknik Tanım

**Denetimsiz öğrenme**, yalnızca **özniteliklere** ($X$) sahip olduğumuz, ancak **etiketlerin** ($y$) bulunmadığı bir makine öğrenimi türüdür.

Temel amaç, bilinen bir sonucu tahmin etmek değil, veri içerisindeki **gizli örüntüleri**, **gruplandırmaları** (kümelemeyi) veya **içsel yapıları** ortaya çıkarmaktır.

Bu, şu soruyu sormak gibidir:
👉 "Bilgisayar, benim ne aradığımı söylememe gerek kalmadan bu kaostaki bir düzeni bulabilir mi?"

### Örnek Senaryo: Meyve Ayırma
Bir sepette hem karpuz hem de elma olduğunu düşünün, ancak bunlar etiketlenmemiş. Denetimsiz öğrenme algoritması, tüm karpuzlar ve tüm elmalar arasında ortak olanı bulmanın bir yolunu bulacaktır. Bu, boyut, renk, şekil ve diğer ayırt edici niteliklere dayanabilir. Algoritma, bu niteliklere göre verileri **kümelendirerek** ayırma işlemini gerçekleştirecektir.

### Örnek Senaryo: Kütüphane Organizasyonu
Bir kütüphanedeki büyük bir kitap yığınını düşünün; kitaplarda kategori veya yazar etiketi yok. Denetimsiz bir öğrenme algoritması, kaosu düzene sokmak için kapak rengi, sayfa uzunluğu veya hatta kullanılan kağıt türüne göre kitapları gruplandırarak başlayabilir.

***

## 3. Neden Veriyi Etiketlemiyoruz?

Denetimsiz öğrenme algoritmaları, insan tarafından sağlanmış etiketleri veya etiketlemeyi kullanmaz. Peki bu neden önemlidir?

Bunun en büyük nedeni, veriyi **etiketlemenin** genellikle **zor, pahalı** veya hatta **imkansız** olmasıdır.

### Etiketleme (Veri Açıklaması - Data Annotation) Nedir?
Makine öğreniminde, veri etiketleme, ham verinin (tablodaki satırlar, görseller, ses kayıtları vb.) alınıp buna bir veya daha fazla etiket atanması anlamına gelir. Bu etiketler bağlam ve anlam sağlayarak makine öğrenimi modelinin öğrenmesini sağlar.

### Zorluklar
1.  **Maliyet ve Zaman:** Tıbbi araştırma gibi alanlarda o kadar çok veri vardır ki, her şeyi etiketlemek yıllar sürebilir ve uzmanlardan muazzam bir çaba gerektirir.
2.  **Bilgi Eksikliği:** Genetik veya astronomi gibi alanlarda, bazen ne aradığımızı bile henüz bilmiyoruz. Eğer fenomeni anlamıyorsak, onu nasıl etiketleyebiliriz?

İşte denetimsiz öğrenme bu noktada parlar: Ham, etiketsiz verideki **yapıyı keşfetmemize** yardımcı olur.

### Uygulama Alanları
| Uygulama | Açıklama | Denetimsiz Yöntem |
| :--- | :--- | :--- |
| **Müşteri Segmentasyonu** | Bir şirketin milyonlarca satın alma kaydı olabilir ancak müşteri tipleri için net etiketleri yoktur. Algoritma, insanları "bütçe odaklı" veya "sadık müşteriler" olarak manuel etiketlemek yerine, verideki doğal gruplandırmaları bulur. | **Kümeleme (Clustering)** |
| **Doğal Dil İşleme** | Temaları bilinmeyen büyük bir metin koleksiyonunu analiz etmek. Denetimsiz öğrenme, önceden tanımlanmış kategorilere ihtiyaç duymadan belgeleri konularına, kelime örüntülerine veya yazım stillerine göre gruplandırabilir. | **Konu Modellemesi (Topic Modeling)** |
| **Boyut Azaltma** | Çok fazla özniteliğe sahip verinin (örneğin binlerce) temel, daha az sayıda değişkene indirgenmesi. | **Temel Bileşen Analizi (PCA)** |


# 🌍 Gerçek Dünya Denetimsiz Öğrenme Örnekleri (Real-World Examples of Unsupervised Learning)

Denetimsiz öğrenme, etiketsiz veriden doğal yapıları, örüntüleri ve ilişkileri keşfetme yeteneği sayesinde birçok sektörde kritik rol oynamaktadır. İşte en yaygın ve etkili uygulamalardan bazıları:

| Alan (Domain) | Uygulama Adı ve Amacı | Temel Denetimsiz Teknikler | Nasıl Çalışır (Denetimsiz Doğa) |
| :--- | :--- | :--- | :--- |
| **Pazarlama ve E-Ticaret** 🛍️ | **Müşteri Segmentasyonu (Customer Segmentation)**: Müşterileri ortak özelliklerine (satın alma geçmişi, demografi, gezinme davranışı) göre gruplamak. | **Kümeleme (Clustering)**: K-Means, DBSCAN, Hiyerarşik Kümeleme. | Algoritma, hangi müşteri gruplarının var olduğunu *önceden bilmez*. Veriyi analiz ederek doğal olarak benzer davranış gösteren grupları **otomatik olarak keşfeder**. |
| **Finans ve Siber Güvenlik** 🛡️ | **Anormallik/Sahtekarlık Tespiti (Anomaly/Fraud Detection)**: Normal örüntüden belirgin şekilde sapan sıra dışı veri noktalarını (işlemler, ağ trafiği) belirlemek. | **Kümeleme (Clustering)**: DBSCAN; **Boyut Azaltma (Dimensionality Reduction)**: Autoencoders, Isolation Forest. | Algoritma, verideki 'normal' davranış modelini öğrenir. Yeni bir işlem bu normal modelin **dışına çıktığında** onu anomali olarak işaretler, yani anormali etiketleyen bir **eğitmen yoktur**. |
| **Perakende ve Lojistik** 🛒 | **Pazar Sepeti Analizi (Market Basket Analysis)**: Müşterilerin birlikte satın alma olasılığı en yüksek olan ürün gruplarını bulmak. | **Birliktelik Kuralı Madenciliği (Association Rule Mining)**: Apriori, Eclat. | Sistem, "X ürünü satın alındıysa, Y ürünü de satın alınır" gibi **gizli ilişkileri** bulmak için milyonlarca işlemi tarar. Bu ilişkiler **önceden etiketlenmemiştir**. |
| **Tavsiye Sistemleri** 🎬 | **İçerik/Ürün Önerileri (Recommendation Systems)**: Kullanıcının geçmiş davranışına göre diğer kullanıcılarla (Collaborative Filtering) veya ürünlerle (Content-Based Filtering) olan benzerliğini kullanarak öneri sunmak. | **Boyut Azaltma (Dimensionality Reduction)**: SVD, Matris Ayrıştırma (Matrix Factorization). | Model, kullanıcı ve öğe etkileşimlerinin karmaşık yapısını daha düşük boyutlu bir gösterime indirir. Bu, **etiketsiz etkileşim verisinden** kullanıcı tercihlerini **çıkarır**. |
| **Doğal Dil İşleme (NLP)** 📰 | **Konu Modellemesi (Topic Modeling)**: Büyük bir metin koleksiyonundaki (haber makaleleri, e-postalar) ana, gizli konuları otomatik olarak çıkarmak. | **Konu Modellemesi**: Latent Dirichlet Allocation (LDA), NMF. | Algoritma, belgeleri gruplandırmak için **konu etiketlerine** sahip değildir. Metinlerdeki kelimelerin **birlikte geçme örüntülerine** bakarak konuları **kendisi tanımlar**. |
| **Görüntü/Video İşleme** 🖼️ | **Görüntü Sıkıştırma (Image Compression)**: Görüntünün kalitesini kaybetmeden boyutunu azaltmak (veri boyutunu düşürmek). | **Boyut Azaltma (Dimensionality Reduction)**: Temel Bileşen Analizi (PCA). | Model, görüntüdeki en fazla bilgiyi taşıyan temel öznitelikleri (boyutları) bularak veriyi sıkıştırır. Bu, **görüntü etiketleri olmadan** verinin içsel yapısını analiz etmeye dayanır. |
| **Genetik ve Tıp** 🧬 | **Gen İfadesi Verilerinin Kümelenmesi (Gene Expression Data Clustering)**: Farklı genetik bozuklukları veya hücre tiplerini temsil eden gen ifadesi örüntülerini gruplamak. | **Kümeleme (Clustering)**: Hiyerarşik Kümeleme, K-Means. | Araştırmacılar, hangi bozuklukların var olduğunu **bilmediklerinde** algoritma, veri setinde doğal olarak oluşan genetik alt grupları (örneğin, yeni bir kanser alt tipi) keşfeder. |

# Denetimsiz Öğrenme Neden Önemlidir? (Why Unsupervised Learning Matters) 💡

Denetimsiz öğrenme, etiketler olmadan veriyi anlamlandırmamıza olanak tanıdığı için hayati öneme sahiptir. İşte neden önemli olduğuna dair temel nedenler:

## Denetimsiz Öğrenmeye Neden İhtiyaç Duyarız?

### Keşifsel Analiz (Exploratory Analysis) 🧐
Bir veri kümesi hakkında çok az şey bildiğimizde, denetimsiz öğrenme **gizli ilişkileri, örüntüleri veya grupları** keşfetmemize yardımcı olur.
* **Örnek:** Satış verilerinde müşterilerin doğal kümelerini bulmak.

### Büyük, Etiketsiz Veriyle Başa Çıkma 💪
Sosyal medya gönderileri, sağlık kayıtları, finansal işlemler gibi verilerin çoğu etiketsizdir ve manuel olarak etiketlemek için çok büyüktür.
* Denetimsiz algoritmalar bu devasa veri kümelerini düzenleyebilir ve basitleştirebilir.

### Gizli İçgörülerin Kilidini Açma 🔑
Bazen en değerli bulgular, hiç beklemediğimiz bulgulardır.
* Denetimsiz öğrenme, aramayı bile bilmediğimiz **anormallikleri, eğilimleri** ve **yapıları** ortaya çıkarabilir.

---

## Denetimsiz Öğrenmedeki Zorluklar (Challenges) 🚧

Güçlü olmasına rağmen, denetimsiz öğrenme kusursuz değildir. Temel zorluklar şunlardır:

* **Sonuçları Yorumlama:** Etiketler olmadan, kümelerin veya örüntülerin gerçekten anlamlı olup olmadığına karar vermek insanlara kalmıştır.
* **Açık Bir Değerlendirme Metriği Yok:** Denetimli öğrenmenin aksine, karşılaştırma yapabileceğimiz **temel gerçek** (ground truth) bulunmaz.
* **Belirsizlik (Ambiguity):** Farklı algoritmalar (veya aynı algoritmanın farklı çalıştırma sonuçları) biraz farklı gruplandırmalar üretebilir.
* **Hesaplama Karmaşıklığı (Computational Complexity):** Büyük veri kümelerini işlemek çok fazla kaynak gerektirebilir.

---

## Özet 🎉

* **Denetimsiz Öğrenme = Etiket Yok.**
* Amaç, verideki **örüntüleri, grupları veya yapıları** bul

# Denetimli Öğrenmeden Temel Farklar (Key Differences From Supervised Learning) ⚖️

Önceki derste, modellerin etiketlenmemiş veri üzerinde çalışarak anlamlı örüntüler çıkardığı **Denetimsiz Öğrenme** kavramını tanıttık. Şimdi, bu yaklaşımın daha önce öğrendiğiniz **Denetimli Öğrenme** ile nasıl karşılaştırıldığını inceleyelim.

---

## 1. Veri Tipi: Etiketli vs. Etiketsiz Veri

Denetimli ve denetimsiz öğrenme arasındaki en önemli fark, çalıştıkları **veri türüdür**.

| Özellik | Denetimli Öğrenme (Supervised) 🎯 | Denetimsiz Öğrenme (Unsupervised) 🔍 |
| :--- | :--- | :--- |
| **Gereken Veri** | **Etiketli Veri** (Labeled Data) | **Etiketsiz Veri** (Unlabeled Data) |
| **Veri Noktası Yapısı** | Her veri noktası, tahmin edilecek karşılık gelen bir **çıktı/hedef** ($y$) içerir. | Model, önceden "doğru" cevabı bilmeden örüntü veya yapı bulmalıdır. |

### Görsel Karşılaştırma: Veri Görünümü 📊

| Etiketli Veri (Denetimli) | Etiketsiz Veri (Denetimsiz) |
| :---: | :---: |
| Farklı renklerle ayrılmış "kedi" ve "köpek" gibi sınıfların olduğu veri noktaları. Bu, modelin belirli kategorilerden öğrenmesine olanak tanır. | Tüm noktaların aynı renkte olduğu veri noktaları. Bu, denetimsiz modellerin etiketler olmadan örüntü keşfetme zorluğunu temsil eder. |
|  |  |

---

## 2. Nihai Hedef: Tahmin vs. Keşif

Dikkate alınması gereken bir diğer önemli husus da, elimizdeki verilere göre ulaşmak istediğimiz **nihai hedeftir**.

| Hedef | Denetimli Öğrenme (Prediction) ➡️ | Denetimsiz Öğrenme (Discovery) 🗺️ |
| :--- | :--- | :--- |
| **Temel Amaç** | Etiketli verilere dayanarak **çıktıları tahmin etmektir**. Model, girdi ($X$) ile bilinen çıktı ($y$) arasındaki **eşlemeyi (mapping)** öğrenir. | Veri içindeki **gizli yapıları** keşfetmektir (örn. kümeleri bulmak veya boyut sayısını azaltmak). |

### Görsel Örnek: Amaç Netliği

| Denetimli Görev: Ev Fiyatı Tahmini (Regresyon) | Denetimsiz Görev: Müşteri Kümelemesi (Clustering) |
| :---: | :---: |
| Sol alt grafik, ev büyüklüğü ile tahmin edilen fiyat arasındaki ilişkiyi gösteren kırmızı bir regresyon doğrusu içerir. Bu, net bir iş hedefidir. | Sağ alt grafik, veri noktalarının önceden etiket olmadan kümelere ayrıldığını gösterir. Farklı renkler, keşfedilen müşteri segmentlerini temsil eder. |
|  |  |

---

## 3. Problem Türleri ve Algoritmalar

Her iki yaklaşımın ele alabileceği problemler doğası gereği farklıdır.

* **Denetimli Öğrenme:** Genellikle **Sınıflandırma** (Classification) ve **Regresyon** (Regression) problemlerine uygulanır. Hedef, belirli etiketleri veya değerleri tahmin etmektir (örn. Spam/Spam Değil).
* **Denetimsiz Öğrenme:** **Kümeleme** (Clustering), **Boyut Azaltma** (Dimensionality Reduction) ve **Birliktelik Kuralı Madenciliği** (Association Rule Mining) gibi problem türleri için kullanılır.

### Görsel İllüstrasyon: Görev Farklılıkları

| Denetimli Görev: Sınıflandırma | Denetimsiz Görev: Kümeleme ve Boyut Azaltma |
| :---: | :---: |
| Veri noktaları iki ayrı kategoriye (örn. spam vs. spam olmayan e-postalar) ayrılır. | Verinin boyutunu azaltmak ve noktaları farklı kümelere ayırmak için **Temel Bileşen Analizi (PCA)** kullanılarak kümeleme görevi gösterilir. |
|  |  |

---

## 4. Model Karmaşıklığı ve Değerlendirme

### Model Karmaşıklığı

| Özellik | Denetimli Öğrenme (Supervised) 🧠 | Denetimsiz Öğrenme (Unsupervised) 🧩 |
| :--- | :--- | :--- |
| **Tipik Modeller** | Girdi ile çıktı arasında kesin eşlemeler öğrenmesi gereken **Derin Sinir Ağları** (Neural Networks) veya **Rastgele Ormanlar** (Random Forests) gibi daha **karmaşık** modeller içerir. | Amaç örüntü keşfi olduğu için genellikle **K-Ortalamalar** (K-means) veya **PCA** gibi daha **basit** modeller kullanılır. |

### Değerlendirme Metrikleri (Evaluation Metrics)

Modelimizin ne kadar iyi performans gösterdiğini bilmek için kullandığımız metrikler de farklıdır:

| Özellik | Denetimli Öğrenme (Supervised) 💯 | Denetimsiz Öğrenme (Unsupervised) ❓ |
| :--- | :--- | :--- |
| **Metrikler** | Performans; **Doğruluk** (Accuracy), **Hassasiyet** (Precision), **Geri Çağırma** (Recall) veya **RMSE** (Regresyon için Hata Karelerinin Karekökü) gibi **temel gerçeğe** dayalı metriklerle değerlendirilir. | Önceden tanımlanmış etiketler olmadığı için **Siluet Skoru** (Silhouette Score - kümelerin ne kadar iyi ayrıldığını gösterir) gibi **içsel (intrinsic)** metrikler kullanılır. | 

# Denetimli ve Denetimsiz Öğrenme Karşılaştırması 🆚

Bu tablo, makine öğreniminin iki ana paradigması olan Denetimli ve Denetimsiz Öğrenme arasındaki temel teknik farklılıkları detaylı olarak göstermektedir.

| Karşılaştırma Kriteri | Denetimli Öğrenme (Supervised Learning) 🎯 | Denetimsiz Öğrenme (Unsupervised Learning) 🔍 |
| :--- | :--- | :--- |
| **Girdi Verisi (Input Data)** | **Etiketli Veri** (Labeled Data). Veri kümesi, girdi ($X$) ve karşılık gelen çıktı ($y$) hedefleri içerir. | **Etiketsiz Veri** (Unlabeled Data). Veri kümesi yalnızca girdi özniteliklerini ($X$) içerir; çıktı hedefi ($y$) bilinmez. |
| **Temel Amaç (Goal)** | Girdi ile çıktı arasındaki eşlemeyi (mapping) öğrenerek **çıktıyı tahmin etmek** (Prediction). | Veri içindeki **gizli yapıları, örüntüleri veya grupları keşfetmek** (Discovery). |
| **Öğrenme Süreci** | Model, etiketleri bir **öğretmen** (insan uzmanı veya önceden belirlenmiş etiket) gibi kullanarak girdi-çıktı ilişkisini öğrenir. | Model, herhangi bir rehberlik olmadan verinin içsel özelliklerine (benzerlik, farklılık vb.) dayanarak kendi kendine öğrenir. |
| **Ortak Problemler** | **Sınıflandırma** (Classification: ikili/çok sınıflı) ve **Regresyon** (Regression: sürekli değer tahmini). | **Kümeleme** (Clustering), **Boyut Azaltma** (Dimensionality Reduction), **Birliktelik Kuralı Madenciliği** (Association Rule Mining). |
| **Yaygın Algoritmalar** | Lineer Regresyon, Lojistik Regresyon, Destek Vektör Makineleri (SVM), Karar Ağaçları, Rastgele Ormanlar, Sinir Ağları. | K-Ortalamalar (K-Means), DBSCAN, Hiyerarşik Kümeleme, Temel Bileşen Analizi (PCA), Otomatik Kodlayıcılar (Autoencoders), Birliktelik Kuralı (Apriori). |
| **Modelin Çıktısı** | Belirli bir **etiket** (kategori) veya **sayısal değer** (tahmin). | **Küme ID'leri**, **indirgenmiş boyutlu özellikler** veya **birliktelik kuralları**. |
| **Değerlendirme Metrikleri** | **Temel Gerçeğe** (Ground Truth) dayalıdır: Doğruluk (Accuracy), Kesinlik (Precision), Geri Çağırma (Recall), F1 Skoru, RMSE (Hata Karelerinin Karekökü). | **İçsel Metrikler** kullanılır: Siluet Skoru (Silhouette Score), Küme İçi Kareler Toplamı (Within-Cluster Sum of Squares - WCSS), Rand İndeksi. |
| **Veri Maliyeti** | **Yüksek.** Etiketli veri elde etmek ve korumak zaman alıcı ve pahalıdır. | **Düşük.** Etiketsiz ham veri genellikle daha kolay ve ucuza toplanır. |
| **Risk ve Belirsizlik** | Tahminler genellikle daha **kesin** ve doğrudan iş hedefleriyle ilişkilidir. | Sonuçların (kümelerin) **yorumlanması** insan uzmanlığı gerektirir ve farklı algoritma çalıştırmaları belirsiz sonuçlar üretebilir. |

