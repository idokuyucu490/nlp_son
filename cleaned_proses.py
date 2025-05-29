import pandas as pd
import re

# === Dosyayı yükle ===
df = pd.read_csv("PRESCRIPTIONS.csv")

# === Sütun adlarını normalize et ===
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# === Hangi sütunları dahil etmeyeceğiz? ===
exclude_columns = [
    "row_id", "subject_id", "hadm_id", "icustay_id",
    "startdate", "enddate", "formulary_drug_cd", "gsn", "ndc",
    "dose_val_rx", "form_val_disp"
]

# === Sadece anlamlı sütunları seçip cümle üret ===
def build_clean_sentence(row):
    parts = []
    for col in row.index:
        if col not in exclude_columns:
            val = str(row[col]).strip()
            # Sayı, boşluk, kod gibi olanları at
            if val and not val.lower() in ['nan', '*nf*'] and not re.fullmatch(r'\d+|\d+\.\d+', val):
                val = re.sub(r'[^a-zA-Z0-9\s/()-]', '', val)  # özel karakter temizle
                parts.append(val.lower())
    return " ".join(parts)

df["clean_sentence"] = df.apply(build_clean_sentence, axis=1)

# === Sonuçları göster ===
print("\n📌 İlk 5 işlenmiş cümle:")
print(df[["row_id", "clean_sentence"]].head())

# === Kaydet
df.to_csv("prescriptions_clean_sentences.csv", index=False)
print("\n✅ 'clean_sentence' sütunu oluşturuldu ve 'prescriptions_clean_sentences.csv' dosyasına kaydedildi.")
