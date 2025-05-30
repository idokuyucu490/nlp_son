# nlp_son
# 📚 Reçete Metinleri Üzerinde TF-IDF ve Word2Vec ile Anlamsal Benzerlik Analizi

Bu proje, reçete metinlerinden oluşan bir veri seti üzerinde **TF-IDF** ve **Word2Vec** yöntemleriyle metinler arası **anlamsal benzerlik ölçümü** yapmayı hedefler. Farklı model yapılandırmaları (CBOW, SkipGram, pencere boyutu, vektör boyutu) ile benzerlik başarıları karşılaştırılır ve Jaccard benzerlik matrisi ile sıralama tutarlılığı analiz edilir.

---

## 🔧 Proje İçeriği

- **Veri Seti**:
  - `prescriptions_lemma.csv` ve `prescriptions_stem.csv`
  - Her satır bir reçete metnini içerir.

- **Vektörleştirme Yöntemleri**:
  - **TF-IDF**: Lemmatized ve Stemmed versiyonları ile ayrı ayrı hesaplandı.
  - **Word2Vec**: CBOW ve SkipGram mimarileri, pencere boyutu (2, 4) ve vektör boyutu (100, 300) kombinasyonlarıyla toplam 16 model eğitildi.

- **Benzerlik Ölçümü**:
  - Her model için cosine similarity ile giriş metnine en yakın 5 metin çıkarıldı.
  - Modeller arası sıralama benzerliği için Jaccard benzerlik matrisi oluşturuldu.

---
