import json
import time
import os
import anthropic
from pathlib import Path

# === SETUP ===
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if ANTHROPIC_API_KEY is None:
    raise ValueError("ANTHROPIC_API_KEY not set. Bitte exportiere ihn vorher.")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

INPUT_FILE = "data/klexikon_texts_large.jsonl"
OUTPUT_DIR = Path("out")
OUTPUT_DIR.mkdir(exist_ok=True)
CHUNK_SIZE = 500
SYSTEM_PROMPT = "Du bist ein freundlicher Lernbegleiter für 4–10-jährige Kinder. Du erklärst Dinge in einfachen, sicheren und liebevollen Worten."

# === FUNKTIONEN ===
def call_claude(prompt, model="claude-3-sonnet", temperature=0.7):
    while True:
        try:
            response = client.messages.create(
                model=model,
                temperature=temperature,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"[API-Fehler]: {e}\n_ Warte 30 Sekunden und versuche es erneut...")
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

def get_last_title_from_output():
    existing_chunks = sorted(OUTPUT_DIR.glob("dpo_gpt35_chunk_*.jsonl"))
    if not existing_chunks:
        return None

    last_file = existing_chunks[-1]
    with open(last_file, "r", encoding="utf-8") as f:
        last_title = None
        for line in f:
            try:
                entry = json.loads(line)
                last_title = entry.get("metadata", {}).get("prompt_id")
            except Exception:
                continue
    return last_title

def get_next_chunk_index():
    existing = list(OUTPUT_DIR.glob("dpo_gpt35_chunk_*.jsonl"))
    if not existing:
        return 1
    return max([int(f.stem.split("_")[-1]) for f in existing]) + 1

# === DATEN LADEN ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

last_title = get_last_title_from_output()

if last_title:
    print(f"⏩ Letzter verarbeiteter Titel: {last_title}")
    start_index = 0
    for i, item in enumerate(data):
        if item.get("title") == last_title:
            start_index = i
            break
    data = data[start_index:]  # Artikel evtl. doppelt, aber sicher
    print(f"🟢 Weiter bei Artikel: {data[0].get('title')} (Index: {start_index})")
else:
    print("🚀 Starte von Anfang.")

chunk_index = get_next_chunk_index()
dpo_data = []

# === VERARBEITUNG ===
for item in data:
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

# Rest speichern
if dpo_data:
    output_file = OUTPUT_DIR / f"dpo_gpt35_chunk_{chunk_index:03}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dpo_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✅ Finale Speicherung: {output_file} mit {len(dpo_data)} Einträgen")
