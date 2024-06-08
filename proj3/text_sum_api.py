from fastapi import FastAPI, Form
# step1 import modules
from transformers import pipeline

# step2 create inference instance
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")

app = FastAPI()


@app.post("/text/")
async def text(text: str = Form()):

    # step4 inference
    result = summarizer(text)

    return {"result": result}

# uvicorn text_sum_api:app