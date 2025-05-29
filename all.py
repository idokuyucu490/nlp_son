import pandas as pd
import glob
import os

# results klasöründeki tüm sonuç dosyaları
result_files = glob.glob("results/results_*.csv")

# Model isimlerini ve dosyalarını tutan sözlük
model_results = {}

for filepath in result_files:
    # Dosya adından model bilgisini çıkar (örn: results_lemmatized_cbow_win2_dim100.csv)
    model_name = os.path.basename(filepath).replace("results_", "").replace(".csv", "")
    
    # CSV oku
    df = pd.read_csv(filepath)
    
    # En benzer 5 sonucu al
    top5 = df.sort_values("rank").head(5).copy()
    
    # Skorları ve metinleri listeye al
    scores = top5["cosine_score"].tolist()
    texts = top5["text"].tolist()
    
    # Model sonuçlarını sözlüğe ekle
    model_results[model_name] = {
        "scores": scores,
        "texts": texts
    }

# Ekrana yazdırma ve CSV'ye yazma için liste hazırla
rows = []
for model_name, res in model_results.items():
    print(f"Model: {model_name}")
    print("5 Benzer Metin:", res['texts'])
    print("Skorlar:", res['scores'])
    print()

    rows.append({
        "model_name": model_name,
        "top5_texts": " | ".join(res['texts']),
        "top5_scores": ", ".join(f"{score:.5f}" for score in res['scores']),
        "average_score": sum(res['scores'])/len(res['scores'])
    })

# DataFrame oluştur ve CSV'ye kaydet
df_summary = pd.DataFrame(rows, columns=["model_name", "top5_texts", "top5_scores", "average_score"])
df_summary.to_csv("model_top5_summary.csv", index=False)

print("✅ model_top5_summary.csv dosyasına özet başarıyla kaydedildi.")
