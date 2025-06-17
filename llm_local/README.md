# 🦁 LLM Local – DPO-Daten mit LeoLM generieren

Dieses Modul dient dazu, aus thematischen Texten (z. B. aus Klexikon) automatisch kindgerechte DPO-Daten zu erzeugen. Es verwendet ein lokales LeoLM-Modell (z. B. `LeoLM/Mistral-7B`) via Transformers.

## Struktur
- `scripts/generate_prompt.py`: Erstellt passende Fragen aus Themen.
- `scripts/generate_dpo_batch.py`: Nutzt LeoLM, um `prompt + chosen + rejected` Sätze zu erzeugen.
- `scripts/inference_model.py`: Lädt LeoLM-Modell via Transformers.
- `data/`: Enthält z. B. `klexikon_texts.jsonl`.
- `out/`: Zielordner für den finalen DPO-Datensatz (`.jsonl`).

## Ausführung
```bash
python3 llm_local/scripts/generate_dpo_batch.py

