import json
import time
import os
import anthropic
from pathlib import Path

# === SETUP ===
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

INPUT_FILE = "data/klexikon_texts_large.jsonl"
OUTPUT_DIR = Path("out")
OUTPUT_DIR.mkdir(exist_ok=True)
CHUNK_SIZE = 100
SYSTEM_PROMPT = "Du bist ein freundlicher Lernbegleiter für 4–10-jährige Kinder. Du erklärst Dinge in einfachen, sicheren und liebevollen Worten."
MODEL_NAME = "claude-sonnet-4-20250514"  # Aktives, nicht veraltetes Modell

# === HILFSFUNKTIONEN ===
def call_claude(prompt, temperature=0.7):
    while True:
        try:
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=1024,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            if isinstance(response.content, list):
                # Extrahiere Text aus Claude-Antwort
                joined = "".join([part.text for part in response.content if hasattr(part, "text")])
                return joined.strip()
            else:
                return str(response.content).strip()
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


def get_last_title():
    files = sorted(OUTPUT_DIR.glob("dpo_gpt35_chunk_*.jsonl"))
    if not files:
        return None
    last_file = files[-1]
    with open(last_file, "r", encoding="utf-8") as f:
        last_line = list(f)[-1]
        try:
            obj = json.loads(last_line)
            return obj.get("metadata", {}).get("prompt_id")
        except:
            return None

# === MAIN ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

last_title = get_last_title()
start_index = 0

if last_title:
    for i, item in enumerate(data):
        if item.get("title") == last_title:
            start_index = i  # eventuell +1 wenn du den letzten NICHT doppelt willst
            break
    print(f"__ Letzter verarbeiteter Titel: {last_title}")
    print(f"__ Weiter bei Artikel: {data[start_index]['title']} (Index: {start_index})")
else:
    print("__ Starte von Anfang")

chunk_index = len(list(OUTPUT_DIR.glob("dpo_gpt35_chunk_*.jsonl"))) + 1

print(f"📁 Bereits vorhandene Output-Dateien: {chunk_index - 1}, Starte mit Datei: {chunk_index:03}")
dpo_data = []

for i, item in enumerate(data[start_index:], start=start_index):
    title = item.get("title")
    text = item.get("text")
    if not title or not text:
        continue

    print(f"🔄 Bearbeite Artikel {i+1}/{len(data)}: {title}")

    prompt_text = f"""Erstelle 5 kindgerechte Quizfragen (nur Fragen!) zu folgendem Thema für 4–10-Jährige:\n\nTitel: {title}\n\nText: {text}"""
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
            print(f"  ✅ Frage hinzugefügt: {frage[:60]}...")

    if i % 10 == 0:
        print(f"🧮 Zwischengröße: {len(dpo_data)} DPO-Einträge")

    if len(dpo_data) >= CHUNK_SIZE:
        output_file = OUTPUT_DIR / f"dpo_gpt35_chunk_{chunk_index:03}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for entry in dpo_data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"💾 Zwischenspeicherung: {output_file} mit {len(dpo_data)} Einträgen")
        chunk_index += 1
        dpo_data = []

if dpo_data:
    output_file = OUTPUT_DIR / f"dpo_gpt35_chunk_{chunk_index:03}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dpo_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"✅ Finale Speicherung: {output_file} mit {len(dpo_data)} Einträgen")
else:
    print("⚠️ Keine neuen DPO-Einträge erzeugt.")
