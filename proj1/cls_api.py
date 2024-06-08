import PIL.Image
from fastapi import FastAPI, File, UploadFile

# STEP 1: Import the necessary modules. 모듈가져오기
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object. 추론기 만들기
base_options = python.BaseOptions(model_asset_path='models\\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=3)
classifier = vision.ImageClassifier.create_from_options(options)
# 계속 로드하지 않도록함, 추론 객체는 한 번만 만들어두기
# 결론: 모델 사용을 확실히 해두고, 서버 띄우기

app = FastAPI()

import io
import PIL
import numpy as np

## 이미지 파일 처리할때는 이 방법 추천!
## 이미지 타입 확인하기
@app.post("/uploadfile/") #메타 정보만(ex)파일 이름, 확장자) 만 os에 보냄
async def create_upload_file(file: UploadFile):
    # file.read() #이걸해야 파일이 웹 서버로 넘어옴 -> await 추가하면 비동기 처리 가능한 부분

    byte_file = await file.read()

    # STEP 3: Load the input image. 추론할 데이터 가져오고
    ## 이미지 파일 읽을 수 있게 변경해주기 (외우기!)
    # 1. convert char array to binary array
    image_bin = io.BytesIO(byte_file)

    # 2. create PIL image from binary array
    pil_img = PIL.Image.open(image_bin)

    # 3. Convert MP image from PIL IMAGE
    image = mp.Image(
    image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # image = mp.Image.create_from_file(IMAGE_FILENAMES[1])

    # STEP 4: Classify the input image. 추론된 결과
    classification_result = classifier.classify(image)
    print(classification_result)

    # STEP 5: Process the classification result. In this case, visualize it. 어떻게 보여줄지 
    img_list = []
    for i in range(3):
        answer = classification_result.classifications[0].categories[i]
        img_list.append([answer.category_name, answer.score])

    return {"result": img_list}


#uvicorn cls_api:app