import openai
import jsonlines
from tqdm import tqdm
import yaml
import os
from scripts.utils import make_boring_version

# Lade Config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

openai.api_key = config["openai_api_key"]
model = config["model"]
temperature = config["temperature"]
max_tokens = config["max_tokens"]

INPUT_FILE = "data/klexikon_texts_large.jsonl"
OUTPUT_FILE = "outputs/dpo_dataset.jsonl"

def call_llm(system_prompt, user_prompt):
    try:
        res = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return res['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Fehler bei LLM-Aufruf: {e}")
        return None

system_prompt = "Du bist ein kinderfreundlicher Erklär-Bot für 6–10-Jährige. Antworte einfach, verspielt, liebevoll und korrekt."

with jsonlines.open(INPUT_FILE) as reader, jsonlines.open(OUTPUT_FILE, mode='w') as writer:
    for row in tqdm(reader, desc="Generiere DPO-Beispiele"):
        title = row.get("title", "").strip()
        text = row.get("text", "").strip()
        if not title or not text:
            continue

        question = f"Was weißt du über {title}?"
        good_answer = call_llm(system_prompt, question + "\n\nHier ist ein Lexikontext dazu:\n" + text)
        if not good_answer:
            continue
        bad_answer = make_boring_version(good_answer)

        entry = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": good_answer}
            ],
            "rejected_message": {
                "role": "assistant",
                "content": bad_answer
            },
            "split": "TRAIN",
            "metadata": {
                "prompt_id": title.lower().replace(" ", "_")
            }
        }
        writer.write(entry)
