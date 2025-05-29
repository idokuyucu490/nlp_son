import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 📥 Dosyayı oku ve sütunları küçük harfe çevir
df_stem = pd.read_csv("prescriptions_stem.csv", encoding="utf-8")
df_stem.columns = df_stem.columns.str.strip().str.lower()

# 🔍 Metin sütununu otomatik bul
text_col = None
for col in df_stem.columns:
    if "stem" in col or "clean" in col:
        text_col = col
        break

if text_col is None:
    raise ValueError("Metin sütunu bulunamadı. Örn: 'stem_sentence' gibi.")

# 📊 TF-IDF vektörleri
tfidf_stem = pd.read_csv("tfidf_stemmed.csv", index_col=0)

# 🎯 Giriş metni index'i (örnek: 9)
selected_index = 9

# 🔁 Benzerlik hesabı
selected_vector = tfidf_stem.iloc[selected_index].values.reshape(1, -1)
similarities = cosine_similarity(selected_vector, tfidf_stem.values)[0]

# 🔝 En benzer 5 metin
top_indices = similarities.argsort()[::-1][1:6]

# 📤 Çıktı
print("\n📌 Giriş metni:")
print(df_stem[text_col].iloc[selected_index])
print("\n📊 En benzer 5 metin ve benzerlik skorları:\n")
for rank, idx in enumerate(top_indices, 1):
    sim_score = round(similarities[idx], 5)
    print(f"{rank}. (index: {idx}) | Score: {sim_score} | Text: {df_stem[text_col].iloc[idx]}")
