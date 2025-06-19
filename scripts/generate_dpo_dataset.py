import json
from pathlib import Path
from transformers import pipeline

# 🔧 Modellpipeline vorbereiten
generator = pipeline("text-generation", model="LeoLM/leo-mistral-hessianai-7b", device_map="auto")

# 🔧 Pfade definieren
INPUT_FILE = "data/klexikon_texts_test.jsonl"
OUTPUT_FILE = "data/klexikon_dpo_dataset.jsonl"

# 🔧 Systemprompt für kindgerechte Assistenz
SYSTEM_PROMPT = "Du bist ein freundlicher Lernbegleiter für Kinder von 4–10 Jahren."

# 🔧 Funktion zur Generierung von Einträgen im DPO-Format
def generate_dpo_entries(title, text, num_examples=5):
    prompt = (
        f"Lies den folgenden Kindertest über das Thema '{title}'. "
        f"Erstelle daraus {num_examples} kindgerechte Fragen für 4–10-jährige Kinder. "
        f"Gib zu jeder Frage eine gute Antwort (kindgerecht, sachlich, freundlich) und eine schlechte Antwort (nicht hilfreich, falsch, übertrieben oder zu technisch). "
        f"Antworte im JSON-Format als Liste mit Objekten mit den Schlüsseln: question, good_answer, bad_answer.\n"
        f"\nTEXT:\n{text}\n"
    )

    result = generator(prompt, max_new_tokens=1024, do_sample=False)[0]["generated_text"]

    try:
        json_start = result.index("[")
        json_data = json.loads(result[json_start:])
    except Exception as e:
        print(f"❌ Fehler bei JSON-Parsing für '{title}':", e)
        return []

    examples = []
    for idx, item in enumerate(json_data):
        try:
            entry = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["good_answer"]}
                ],
                "rejected_message": {"role": "assistant", "content": item["bad_answer"]},
                "split": "TRAIN",
                "metadata": {"prompt_id": title.replace(" ", "_"), "index": idx}
            }
            examples.append(entry)
        except Exception as e:
            print(f"⚠️ Fehler bei Eintrag {idx} für '{title}':", e)

    return examples

# 🔄 Verarbeitung starten
with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    for line in infile:
        article = json.loads(line)
        title = article.get("title", "Thema")
        text = article.get("text", "")
        print(f"🔍 Generiere Fragen für: {title}")
        entries = generate_dpo_entries(title, text)
        for entry in entries:
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ Fertig. Datei gespeichert unter: {OUTPUT_FILE}")
