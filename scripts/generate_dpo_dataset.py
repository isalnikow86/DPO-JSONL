import json
import random
import time
import openai
import os
from pathlib import Path

# === SETUP ===
openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    raise ValueError("OPENAI_API_KEY not set. Please export it before running the script.")

INPUT_FILE = "data/klexikon_texts_large.jsonl"
OUTPUT_FILE = "out/dpo_gpt35_output.jsonl"

SYSTEM_PROMPT = "Du bist ein freundlicher Lernbegleiter für 4–10-jährige Kinder. Du erklärst Dinge in einfachen, sicheren und liebevollen Worten."

# === FUNKTIONEN ===
def call_chatgpt(prompt_text, retries=3):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.7,
                max_tokens=700
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("Fehler bei LLM-Aufruf:", e)
            time.sleep(2)
    return ""

def build_dpo_entry(prompt, good, bad, prompt_id):
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": good}
        ],
        "rejected_message": {"role": "assistant", "content": bad},
        "split": "TRAIN",
        "metadata": {"prompt_id": prompt_id}
    }

# === START ===
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

Path("out").mkdir(parents=True, exist_ok=True)
dpo_data = []

for item in data:
    title = item.get("title")
    text = item.get("text")
    if not title or not text:
        continue

    # Frage-Generierungs-Prompt korrekt mit mehrzeiligem f-String
    prompt_text = f"""Erstelle 5 kindgerechte Quizfragen (nur Fragen!) zu folgendem Thema für 4–10-Jährige:

Titel: {title}

Text: {text}
"""

    frage_block = call_chatgpt(prompt_text)
    if not frage_block:
        continue

    fragen = [line.strip("- ").strip() for line in frage_block.split("\n") if line.strip() and "?" in line]

    for idx, frage in enumerate(fragen[:5]):
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
            dpo_data.append(build_dpo_entry(frage, good_answer.strip(), bad_answer.strip(), prompt_id=title))


# === SPEICHERN ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for entry in dpo_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ Fertig! {len(dpo_data)} DPO-Beispiele gespeichert in: {OUTPUT_FILE}")
