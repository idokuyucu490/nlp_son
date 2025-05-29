import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Dosyayı oku ve sütunları düzelt
df_lemma = pd.read_csv("prescriptions_lemma.csv", encoding="utf-8")
df_lemma.columns = df_lemma.columns.str.strip().str.lower()

# Sütun adını kontrol et
print("Sütunlar:", df_lemma.columns.tolist())

# TF-IDF verisini oku
tfidf_lemma = pd.read_csv("tfidf_lemmatized.csv", index_col=0)

# Giriş metni index'i
selected_index = 9

# Vektörünü al
selected_vector = tfidf_lemma.iloc[selected_index].values.reshape(1, -1)

# Benzerlik hesapla
similarities = cosine_similarity(selected_vector, tfidf_lemma.values)[0]
tops_idx = similarities.argsort()[::-1][1:6]

# Giriş metni sütunu adını güncel sütunlardan al
target_col = "stem_sentence" if "stem_sentence" in df_lemma.columns else df_lemma.columns[1]

print("Giriş metni:", df_lemma[target_col].iloc[selected_index])
print("\nEn benzer 5 başlık ve skorları:")
for i, idx in enumerate(tops_idx, 1):
    print(f"{i}. ({idx}) {df_lemma[target_col].iloc[idx]} --> {similarities[idx]:.4f}")
