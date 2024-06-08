from fastapi import FastAPI, Form
# step1 import modules
from transformers import pipeline

# step2 create inference instance
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

app = FastAPI()


@app.post("/text/")
async def text(text: str = Form()):

    # # step3 prepare input data
    # text = "부품 공급 차질에…기아차 광주공장 전면 가동 중단"

    # step4 inference
    result = classifier(text)

    # # step5 print
    # print(result)

    return {"result": result}

# uvicorn text_cls_api:app