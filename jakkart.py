import pandas as pd
import glob
import os

# Sonuç dosyalarının yolu
result_files = glob.glob("results/results_*.csv")

# Model isimleri ve top 5 ID setlerini tutan sözlük
model_top5_ids = {}

for filepath in result_files:
    model_name = os.path.basename(filepath).replace("results_", "").replace(".csv", "")
    df = pd.read_csv(filepath)
    
    top5_ids = set(df[df['rank'] <= 5]['id'].astype(str).tolist())  # ID'leri string olarak al
    
    model_top5_ids[model_name] = top5_ids

# Jaccard similarity hesaplayan fonksiyon
def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0:
        return 0
    return len(intersection) / len(union)

# Model isimlerine göre Jaccard matrisi oluştur
models = list(model_top5_ids.keys())
n = len(models)

jaccard_matrix = pd.DataFrame(index=models, columns=models, dtype=float)

for i in range(n):
    for j in range(n):
        if i == j:
            jaccard_matrix.iloc[i, j] = 1.0
        else:
            set1 = model_top5_ids[models[i]]
            set2 = model_top5_ids[models[j]]
            jaccard_matrix.iloc[i, j] = jaccard_similarity(set1, set2)

# Sonucu CSV'ye kaydet
jaccard_matrix.to_csv("jaccard_similarity_matrix.csv")

print("Jaccard similarity matrisi 'jaccard_similarity_matrix.csv' olarak kaydedildi.")
