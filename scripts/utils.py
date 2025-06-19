import os
import openai

openai.api_key = os.environ.get("OPENAI_API_KEY")

def make_boring_version(text: str) -> str:
    """
    Sachliche, emotionslose Version erzeugen (nicht kindgerecht).
    """
    prompt = f"""
    Formuliere diesen Text sachlich, emotionslos und nüchtern – kindgerecht, aber ohne Begeisterung:

    {text}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du schreibst sachlich und emotionslos."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print("Fehler bei LLM-Aufruf:", e)
        return ""

def make_fun_version(text: str) -> str:
    """
    Freundlich-kindgerechte Version erzeugen (spielerisch, liebevoll).
    """
    prompt = f"""
    Erkläre diesen Text freundlich, liebevoll und kindgerecht für Kinder von 4–10 Jahren:

    {text}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein freundlicher Lernbegleiter für Kinder."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print("Fehler bei LLM-Aufruf:", e)
        return ""
