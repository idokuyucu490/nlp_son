import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# === Dosyaları yükle
df_lemma = pd.read_csv("prescriptions_lemma_only.csv")
df_stem = pd.read_csv("prescriptions_stem_only.csv")

# === Sütun adlarını normalize et
df_lemma.columns = [col.strip().lower() for col in df_lemma.columns]
df_stem.columns = [col.strip().lower() for col in df_stem.columns]

# === TF-IDF vektörizer tanımı
vectorizer = TfidfVectorizer()

# === Lemmatized cümleler için TF-IDF
tfidf_lemma = vectorizer.fit_transform(df_lemma["lemma_sentence"])
df_tfidf_lemma = pd.DataFrame(tfidf_lemma.toarray(), columns=vectorizer.get_feature_names_out())
df_tfidf_lemma.insert(0, "row_id", df_lemma["row_id"])

# === Stemmed cümleler için TF-IDF
tfidf_stem = vectorizer.fit_transform(df_stem["stem_sentence"])
df_tfidf_stem = pd.DataFrame(tfidf_stem.toarray(), columns=vectorizer.get_feature_names_out())
df_tfidf_stem.insert(0, "row_id", df_stem["row_id"])

# === CSV dosyalarına yaz
df_tfidf_lemma.to_csv("tfidf_lemmatized.csv", index=False)
df_tfidf_stem.to_csv("tfidf_stemmed.csv", index=False)

print("✅ TF-IDF matrisleri başarıyla oluşturuldu ve kaydedildi:")
print(" - tfidf_lemmatized.csv")
print(" - tfidf_stemmed.csv")
