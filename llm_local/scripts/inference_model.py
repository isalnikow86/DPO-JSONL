from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

def load_local_model(model_name="LeoLM/leo-mistral-hessianai-7b"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe
