import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF dosya isimleri
tfidf_files = {
    "lemmatized": "tfidf_lemmatized.csv",
    "stemmed": "tfidf_stemmed.csv"
}

for label, filepath in tfidf_files.items():
    print(f"Processing {label}...")

    # Dosyayı oku, kelimeler index, metin isimleri sütunlarda
    df = pd.read_csv(filepath, index_col=0)

    # Satırlar kelimeler, sütunlar metinler -> Transpose ile metinleri satıra al
    tfidf_matrix = df.T.values

    # Metinler arası cosine similarity hesapla
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Sonuç dataframe olarak oluştur
    sim_df = pd.DataFrame(similarity_matrix, index=df.columns, columns=df.columns)

    # CSV olarak kaydet
    out_csv = f"cosine_similarity_{label}.csv"
    sim_df.to_csv(out_csv)

    print(f"Saved cosine similarity matrix to {out_csv}")

print("All done.")
