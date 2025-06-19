import json
import random
import time
import os
import requests
from pathlib import Path

# === SETUP ===
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set. Please export it before running the script.")

INPUT_FILE = "data/klexikon_texts_large.jsonl"
OUTPUT_DIR = Path("out")
OUTPUT_DIR.mkdir(exist_ok=True)
CHUNK_SIZE = 100
SYSTEM_PROMPT = "Du bist ein freundlicher Lernbegleiter für 4–10-jährige Kinder. Du erklärst Dinge in einfachen, sicheren und liebevollen Worten."

HEADERS = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

CLAUDE_MODEL = "claude-3-sonnet-20240229"

API_URL = "https://api.anthropic.com/v1/messages"

def call_claude(prompt):
    while True:
        try:
            response = requests.post(API_URL, headers=HEADERS, json={
                "model": CLAUDE_MODEL,
                "max_tokens": 1024,
                "temperature": 0.7,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}]
            })
            response.raise_for_status()
            return response.json()["content"][0]["text"].strip()
        
        except requests.exceptions.RequestException as e:
            print(f"[API-Fehler]: {e}")
            print("⚠️ Warte 30 Sekunden und versuche es erneut...")
            time.sleep(30)

def build_dpo_entry(question, good, bad, prompt_id):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": good}
        ],
        "rejected_message": {"role": "assistant", "content": bad},
        "split": "TRAIN",
        "metadata": {"prompt_id": prompt_id}
    }

def get_existing_chunks():
    existing = list(OUTPUT_DIR.glob("dpo_gpt35_chunk_*.jsonl"))
    return sorted(existing)

def get_next_chunk_index():
    existing = get_existing_chunks()
    if not existing:
        return 1
    return max([int(f.stem.split("_")[-1]) for f in existing]) + 1

# === MAIN ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

chunk_index = get_next_chunk_index()
dpo_data = []
processed = (chunk_index - 1) * CHUNK_SIZE
print(f"🧠 Starte bei Artikel {processed+1} von {len(data)}... (Datei {chunk_index:03})")

for i, item in enumerate(data[processed:], start=processed):
    title = item.get("title")
    text = item.get("text")
    if not title or not text:
        continue

    prompt_text = f"""Erstelle 5 kindgerechte Quizfragen (nur Fragen!) zu folgendem Thema für 4–10-Jährige:

Titel: {title}

Text: {text}"""

    frage_block = call_claude(prompt_text)
    fragen = [line.strip("- ") for line in frage_block.split("\n") if line.strip() and "?" in line]

    for frage in fragen[:5]:
        good_prompt = f"Beantworte diese Frage kindgerecht, liebevoll und mit Wissen aus folgendem Text:\n\nFrage: {frage}\n\nText: {text}"
        bad_prompt = f"Gib eine sehr kurze, falsche, sachlich klingende Antwort auf diese Frage:\n{frage}"

        good_answer = call_claude(good_prompt)
        bad_answer = call_claude(bad_prompt)

        if good_answer and bad_answer:
            entry = build_dpo_entry(frage, good_answer, bad_answer, prompt_id=title)
            dpo_data.append(entry)

    if len(dpo_data) >= CHUNK_SIZE:
        output_file = OUTPUT_DIR / f"dpo_gpt35_chunk_{chunk_index:03}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for entry in dpo_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"✅ Zwischenspeicherung: {output_file} mit {len(dpo_data)} Einträgen")
        chunk_index += 1
        dpo_data = []

# Restdaten speichern
if dpo_data:
    output_file = OUTPUT_DIR / f"dpo_gpt35_chunk_{chunk_index:03}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dpo_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✅ Finale Speicherung: {output_file} mit {len(dpo_data)} Einträgen")
