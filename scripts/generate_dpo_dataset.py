import json
import os
import openai
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = """
Du bist ein freundlicher Lernbegleiter für Kinder von 4–10 Jahren. Du erklärst Dinge in einfachen, sicheren und liebevollen Worten.
"""

def load_articles(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def create_prompt(title, text):
    return f"""
Thema: {title}

Hier ist ein kurzer Sachtext für Kinder:
"""
{text.strip()}
"""

Erstelle daraus 5 einfache Fragen für Kinder von 4–10 Jahren mit jeweils:
- einer guten, liebevollen und kindgerechten Antwort
- einer falschen, sachlich klingenden, aber inhaltlich falschen Antwort

Gib die Ausgabe im JSON-Format mit folgender Struktur:
[
  {{"question": "...", "good_answer": "...", "bad_answer": "..."}},
  ...
]
"""

def query_openai_chat(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": prompt.strip()}
        ],
        temperature=0.7
    )
    return response["choices"][0]["message"]["content"]

def build_dpo_entries(title, topic_id, qapairs):
    entries = []
    for idx, item in enumerate(qapairs):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["good_answer"]}
        ]
        rejected = {"role": "assistant", "content": item["bad_answer"]}
        entry = {
            "messages": messages,
            "rejected_message": rejected,
            "split": "TRAIN",
            "metadata": {"topic": title, "prompt_id": f"{topic_id}_{idx}"}
        }
        entries.append(entry)
    return entries

def main():
    input_path = "data/klexikon_articles.jsonl"
    output_path = "data/dpo_klexikon_output.jsonl"
    articles = load_articles(input_path)
    all_entries = []

    for i, article in enumerate(tqdm(articles)):
        prompt = create_prompt(article["title"], article["text"])
        try:
            completion = query_openai_chat(prompt)
            parsed = json.loads(completion)
            dpo_entries = build_dpo_entries(article["title"], f"article{i:04d}", parsed)
            all_entries.extend(dpo_entries)
        except Exception as e:
            print(f"Fehler bei Artikel {article['title']}: {e}")
            continue

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ DPO-Datensatz gespeichert unter: {output_path}")

if __name__ == "__main__":
    main()
