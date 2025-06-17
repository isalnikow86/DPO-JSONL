import jsonlines
import os
from tqdm import tqdm

from llm_local.scripts.inference_model import load_local_model
from llm_local.scripts.generate_prompt import generate_question_from_text

INPUT_FILE = "llm_local/data/klexikon_texts.jsonl"
OUTPUT_FILE = "llm_local/out/dpo_dataset.jsonl"

generator = load_local_model()

with jsonlines.open(INPUT_FILE) as reader, jsonlines.open(OUTPUT_FILE, mode='w') as writer:
    for sample in tqdm(reader, desc="Erzeuge DPO-Beispiele"):
        title = sample.get("title", "").strip()
        content = sample.get("text", "").strip()

        if not title or not content:
            continue

        instruction = generate_question_from_text(title, content)

        prompt = f"{instruction}\n\n{content}"

        response = generator(prompt, max_new_tokens=300, do_sample=True, temperature=0.7)[0]["generated_text"]

        writer.write({
            "instruction": instruction,
            "input": "",
            "output": response.strip()
        })
