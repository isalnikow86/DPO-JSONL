import random

def generate_question_from_text(title: str, content: str) -> str:
    examples = [
        f"Was ist das Besondere an {title}?",
        f"Was macht {title} aus?",
        f"Wozu braucht man {title}?",
        f"Wo leben {title}?",
        f"Was frisst ein {title}?",
        f"Wie leben {title}?",
        f"Was können Kinder über {title} lernen?",
    ]
    return random.choice(examples)
