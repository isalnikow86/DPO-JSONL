import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def make_boring_version(text: str) -> str:
    """
    Generiert eine sachlich-nüchterne Version eines Texts mit wenig Emotionen.
    """
    prompt = f"""
    Formuliere folgenden Text sachlich und emotionslos, ohne Begeisterung, kindgerecht, aber nüchtern:

    {text}

    Antworte direkt mit dem Text.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein sachlicher und emotionsloser Text-Generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("Fehler bei LLM-Aufruf:", e)
        return ""

def make_fun_version(text: str) -> str:
    """
    Generiert eine verspielte, liebevolle und kindgerechte Version eines Texts.
    """
    prompt = f"""
    Erkläre folgenden Text liebevoll, kindgerecht und verspielt. Verwende einfache Sprache und ein freundliches Wesen:

    {text}

    Antworte direkt mit dem Text.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein freundlicher Lernbegleiter für Kinder im Alter von 4–10 Jahren."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print("Fehler bei LLM-Aufruf:", e)
        return ""
