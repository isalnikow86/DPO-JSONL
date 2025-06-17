import json
import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Modell-ID
MODEL_ID = "LeoLM/leo-mistral-hessianai-7b"
SYSTEM_PROMPT = "Du bist ein freundlicher Lernbegleiter für Kinder von 4–10 Jahren."

# Eingabe- und Ausgabe-Dateien
INPUT_FILE = "llm_local/data/klexikon_texts.jsonl"
OUTPUT_FILE = "llm_local/out/dpo_dataset.jsonl"

# LLM vorbereiten
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def make_fun_prompt(text: str) -> str:
    return f"Formuliere eine interessante Frage für Kinder zu folgendem Textauszug:\n{text}"

def make_response(prompt: str, temperature=0.7) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    prompt_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
    result = generator(prompt_text, max_new_tokens=200, temperature=temperature, do_sample=True)
    return result[0]["generated_text"].replace(prompt_text, "").strip()

def main():
    with open(INPUT_FILE, "r") as f:
        entries = [json.loads(line) for line in f]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as out_file:
        for i, entry in enumerate(tqdm(entries, desc="Generiere DPO-Beispiele")):
            text = entry["text"]
            title = entry.get("title", f"sample_{i}")

            # Prompt generieren (Frage)
            question_prompt = make_fun_prompt(text)
            user_prompt = make_response(question_prompt, temperature=0.5)

            # Gute und schlechte Antwort
            chosen = make_response(user_prompt, temperature=0.8)
            rejected = make_response(user_prompt, temperature=0.2)

            # DPO-Format speichern
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": chosen}
                ],
                "rejected_message": {"role": "assistant", "content": rejected},
                "split": "TRAIN",
                "metadata": {"prompt_id": title.replace(" ", "_").lower()}
            }
            out_file.write(json.dumps(example, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
