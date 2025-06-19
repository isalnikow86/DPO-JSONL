import json
import random
import time
import openai
from pathlib import Path

# === SETUP ===
openai.api_key = "sk-..."  # 🔁 Trage hier deinen OpenAI API-Key ein
INPUT_FILE = "data/klexikon_texts_test.jsonl"
OUTPUT_FILE = "out/dpo_gpt35_output.jsonl"

SYSTEM_PROMPT = "Du bist ein freundlicher Lernbegleiter für 4-10-jährige Kinder. Du erklärst Dinge in einfachen, sicheren und liebevollen Worten."

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
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Fehler beim API-Call: {e}. Versuche es erneut...")
            time.sleep(2)
    return None

def create_dpo_entry(user_question, good_answer, bad_answer, topic, split="TRAIN"):
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": good_answer}
        ],
        "rejected_message": {"role": "assistant", "content": bad_answer},
        "split": split,
        "metadata": {"prompt_id": topic.lower().replace(" ", "_"), "topic": topic}
    }

# === MAIN ===

data = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

out_path = Path(OUTPUT_FILE)
out_path.parent.mkdir(exist_ok=True, parents=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for entry in data:
        topic = entry["title"]
        context = entry["text"]

        for i in range(5):
            prompt = f"Lies den folgenden kindgerechten Text:\n\n{context}\n\nErstelle eine einzelne Frage, die ein Kind dazu stellen könnte, sowie eine passende kurze Antwort. Füge auch eine falsche Antwort hinzu, die sich nicht gut eignet. Gib das Ergebnis in folgendem Format aus:\n\nFrage: ...\nGute Antwort: ...\nSchlechte Antwort: ..."
            response = call_chatgpt(prompt)
            if not response:
                continue

            try:
                parts = response.split("\n")
                q = [l for l in parts if l.lower().startswith("frage")][0].split(":", 1)[1].strip()
                good = [l for l in parts if l.lower().startswith("gute")][0].split(":", 1)[1].strip()
                bad = [l for l in parts if l.lower().startswith("schlechte")][0].split(":", 1)[1].strip()
                split = random.choice(["TRAIN", "TEST"])
                dpo = create_dpo_entry(q, good, bad, topic, split)
                out_f.write(json.dumps(dpo, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"❌ Fehler beim Parsen: {e}\nAntwort war: {response}")

print("✅ DPO-Datensatz fertig unter:", OUTPUT_FILE)
