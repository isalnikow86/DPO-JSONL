import os
import json
import openai
import time
import glob

openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_FILE = "data/klexikon.jsonl"
OUTPUT_DIR = "out"
PARTIAL_PREFIX = "output_partial_"
FINAL_FILE = "output_all.jsonl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Vorherige Ergebnisse einlesen
existing_titles = set()
for partial_file in glob.glob(os.path.join(OUTPUT_DIR, f"{PARTIAL_PREFIX}*.jsonl")):
    with open(partial_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                prompt_id = entry.get("metadata", {}).get("prompt_id")
                if prompt_id:
                    existing_titles.add(prompt_id)
            except json.JSONDecodeError:
                continue

# Hilfsfunktionen
def call_chatgpt(prompt, temperature=0.7):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=temperature,
            messages=[
                {"role": "system", "content": "Du bist ein freundlicher Lernbegleiter für Kinder zwischen 4 und 10 Jahren."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Fehler bei ChatGPT-Call: {e}")
        time.sleep(5)
        return None

def build_dpo_entry(question, good, bad, prompt_id, split="TRAIN"):
    return {
        "messages": [
            {"role": "system", "content": "Du bist ein freundlicher Lernbegleiter für Kinder zwischen 4 und 10 Jahren."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": good}
        ],
        "rejected_message": {"role": "assistant", "content": bad},
        "split": split,
        "metadata": {
            "prompt_id": prompt_id
        }
    }

# Input-Daten laden
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Verarbeitung starten
dpo_data = []
batch_size = 100
batch_index = 0
processed_count = 0

for item in data:
    title = item.get("title")
    text = item.get("text")
    if not title or not text or title in existing_titles:
        continue

    prompt_text = f"""Erstelle 5 kindgerechte Quizfragen (nur Fragen!) zu folgendem Thema für 4–10-Jährige:

Titel: {title}

Text: {text}"""

    frage_block = call_chatgpt(prompt_text)
    if not frage_block:
        continue

    fragen = [line.strip("- ").strip() for line in frage_block.split("\n") if "?" in line and len(line.strip()) > 3]

    for frage in fragen[:5]:
        good_prompt = f"""Beantworte diese Frage kindgerecht, liebevoll und mit Wissen aus folgendem Text:

Frage: {frage}

Text: {text}"""
        bad_prompt = f"Gib eine sehr kurze, falsche, sachlich klingende Antwort auf diese Frage:\n{frage}"

        good_answer = call_chatgpt(good_prompt)
        bad_answer = call_chatgpt(bad_prompt)

        if good_answer and bad_answer:
            dpo_data.append(build_dpo_entry(frage, good_answer, bad_answer, prompt_id=title))
            processed_count += 1

    # Zwischenspeichern alle batch_size Artikel
    if processed_count >= batch_size:
        out_path = os.path.join(OUTPUT_DIR, f"{PARTIAL_PREFIX}{batch_index}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for entry in dpo_data:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
        print(f"✅ Gespeichert: {out_path} mit {len(dpo_data)} Einträgen")
        dpo_data = []
        processed_count = 0
        batch_index += 1

# Am Ende alle restlichen schreiben
if dpo_data:
    out_path = os.path.join(OUTPUT_DIR, f"{PARTIAL_PREFIX}{batch_index}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in dpo_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    print(f"✅ Letzte Teildatei geschrieben: {out_path}")

# Optional: Alles in eine große Datei zusammenführen
final_path = os.path.join(OUTPUT_DIR, FINAL_FILE)
with open(final_path, "w", encoding="utf-8") as f_out:
    for partial_file in sorted(glob.glob(os.path.join(OUTPUT_DIR, f"{PARTIAL_PREFIX}*.jsonl"))):
        with open(partial_file, "r", encoding="utf-8") as f_in:
            for line in f_in:
                f_out.write(line)
print(f"✅ Gesamtdatei geschrieben: {final_path}")
