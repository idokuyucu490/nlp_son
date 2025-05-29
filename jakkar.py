import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# VS Code klasöründeki başlıklara uygun model isimleri
model_names = [
    "TF-IDF Lemma", "TF-IDF Stem",
    "lemmatized_cbow_win2_dim100", "lemmatized_cbow_win2_dim300",
    "lemmatized_cbow_win4_dim100", "lemmatized_cbow_win4_dim300",
    "lemmatized_skipgram_win2_dim100", "lemmatized_skipgram_win2_dim300",
    "lemmatized_skipgram_win4_dim100", "lemmatized_skipgram_win4_dim300",
    "stemmed_cbow_win2_dim100", "stemmed_cbow_win2_dim300",
    "stemmed_cbow_win4_dim100", "stemmed_cbow_win4_dim300",
    "stemmed_skipgram_win2_dim100", "stemmed_skipgram_win2_dim300",
    "stemmed_skipgram_win4_dim100", "stemmed_skipgram_win4_dim300"
]

# ⚠️ NOT: Gerçek Jaccard hesaplamak için top5_words listesini kullanman gerekir
# Örnek: {"model_adi": set(["word1", "word2", ...])}
# Aşağıdaki örnekte rastgele değerler yerleştirildi (sen gerçek değerle değiştirebilirsin)

# 🔁 Temsili skor matrisi üret (köşegen 1.0, diğerleri 0.20 / 0.43 / 0.67 gibi sabit değerlerden seçiliyor)
np.random.seed(42)
size = len(model_names)
matrix = np.random.choice([0.20, 0.43, 0.67], size=(size, size), p=[0.4, 0.3, 0.3])
np.fill_diagonal(matrix, 1.0)

# DataFrame'e aktar
jaccard_df = pd.DataFrame(matrix, index=model_names, columns=model_names)

# 🌡️ Isı haritası oluştur
plt.figure(figsize=(18, 14))
sns.heatmap(jaccard_df, annot=True, fmt=".2f", cmap="YlGnBu", square=True, linewidths=0.4, cbar=True)
plt.title("Jaccard Benzerlik Matrisi (Top-5 Kelimeye Göre)", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 💾 CSV'ye kaydetmek istersen
jaccard_df.to_csv("jaccard_similarity_matrix.csv")
