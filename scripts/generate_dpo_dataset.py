import os
import json
import openai
from time import sleep

INPUT_FILE = "data/klexikon_texts_large.jsonl"
OUTPUT_FILE = "out/dpo_gpt35_output.jsonl"
CHUNK_PREFIX = "out/dpo_gpt35_chunk_"
CHUNK_SIZE = 100

openai.api_key = os.getenv("OPENAI_API_KEY")

system_prompt = "Du bist ein freundlicher Lernbegleiter für Kinder zwischen 4 und 10 Jahren. Du erklärst Dinge liebevoll, einfach und mit kindgerechten Worten."

def call_chatgpt(prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Fehler bei API-Aufruf]: {e}")
        sleep(5)
        return None

def build_dpo_entry(question, good, bad, prompt_id):
    return {
        "messages": [
            {"role": "system", "content": "Du bist ein freundlicher Lernbegleiter für Kinder zwischen 4 und 10 Jahren. Du erklärst Dinge liebevoll, einfach und verständlich."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": good}
        ],
        "rejected_message": {"role": "assistant", "content": bad},
        "split": "TRAIN",
        "metadata": {"prompt_id": prompt_id}
    }

# Hauptlogik
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

dpo_data = []
all_data = []
chunk_counter = 0

for idx, item in enumerate(data):
    title = item.get("title")
    text = item.get("text")
    if not title or not text:
        continue

    prompt_text = f"""Erstelle 5 kindgerechte Quizfragen (nur Fragen!) zu folgendem Thema für 4–10-Jährige:

Titel: {title}

Text: {text}
"""

    frage_block = call_chatgpt(prompt_text)
    if not frage_block:
        continue

    fragen = [line.strip("- ").strip() for line in frage_block.split("\n") if "?" in line]

    for frage in fragen[:5]:
        good_prompt = f"""Beantworte diese Frage kindgerecht, liebevoll und mit Wissen aus folgendem Text:

Frage: {frage}

Text: {text}
"""
        bad_prompt = f"""Gib eine sehr kurze, falsche, sachlich klingende Antwort auf diese Frage:

{frage}
"""

        good_answer = call_chatgpt(good_prompt)
        bad_answer = call_chatgpt(bad_prompt)

        if good_answer and bad_answer and len(good_answer) > 5 and len(bad_answer) > 2:
            entry = build_dpo_entry(frage, good_answer, bad_answer, prompt_id=title)
            dpo_data.append(entry)
            all_data.append(entry)

    # Chunk-Datei schreiben
    if (idx + 1) % CHUNK_SIZE == 0:
        chunk_counter += 1
        chunk_path = f"{CHUNK_PREFIX}{chunk_counter:03d}.jsonl"
        with open(chunk_path, "w", encoding="utf-8") as f_chunk:
            for entry in dpo_data:
                f_chunk.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[✓] Zwischenspeicherung: {chunk_path} mit {len(dpo_data)} Einträgen")
        dpo_data = []

# Letzten Chunk speichern, falls < CHUNK_SIZE offen
if dpo_data:
    chunk_counter += 1
    chunk_path = f"{CHUNK_PREFIX}{chunk_counter:03d}.jsonl"
    with open(chunk_path, "w", encoding="utf-8") as f_chunk:
        for entry in dpo_data:
            f_chunk.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[✓] Letzter Zwischenspeicher: {chunk_path}")

# Gesamtausgabe
with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    for entry in all_data:
        f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"[✅] Gesamtausgabe gespeichert: {OUTPUT_FILE} mit {len(all_data)} Einträgen.")
