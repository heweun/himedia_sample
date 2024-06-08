from fastapi import FastAPI, Form
# step1 import modules
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# step2 create inference instance
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")

classifier = pipeline("ner", model=model, tokenizer=tokenizer)

app = FastAPI()


@app.post("/text/")
async def text(text: str = Form()):

    # step4 inference
    result = classifier(text)

    return {"result": result}

# uvicorn token_cls_api:app