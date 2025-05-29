import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# İlk sefer için gerekli olabilir
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")

# === 1. Dosyayı yükle ===
df = pd.read_csv("prescriptions_clean_sentences.csv")
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# === 2. Temizleme fonksiyonu ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)  # sadece harf ve boşluk bırak
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === 3. NLP Araçları ===
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# === 4. Temizle, Lemma ve Stemle ===
df["cleaned"] = df["clean_sentence"].apply(clean_text)

df["lemma_sentence"] = df["cleaned"].apply(
    lambda x: " ".join([lemmatizer.lemmatize(token) for token in word_tokenize(x)])
)

df["stem_sentence"] = df["cleaned"].apply(
    lambda x: " ".join([stemmer.stem(token) for token in word_tokenize(x)])
)

# === 5. Sadece gerekli sütunları seç ve ayrı dosyalara yaz ===
df[["row_id", "lemma_sentence"]].to_csv("prescriptions_lemma_only.csv", index=False)
df[["row_id", "stem_sentence"]].to_csv("prescriptions_stem_only.csv", index=False)

print("✅ Lemmatization ve stemming tamamlandı.")
print("💾 Kaydedilen dosyalar:")
print(" - prescriptions_lemma_only.csv")
print(" - prescriptions_stem_only.csv")
