# step1
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

## onix 설치해서 에러 없애기

# step2
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# step3
#img = ins_get_image('t1')
img = cv2.imread('c1.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('c2.png', cv2.IMREAD_COLOR)

# step4
faces1 = app.get(img)
faces2 = app.get(img2)
# print(f"faces:{faces1, faces2}")

# step5
# then print all-to-all face similarity
feats = []
feats.append(faces1[0].normed_embedding) #임베딩 방법 외우기
feats.append(faces2[0].normed_embedding)

feats = np.array(feats, dtype=np.float32)
sims = np.dot(feats[0], feats[1].T)
print(f"\n0과 1사이 유사도 {sims}") #임계점 확인 -- 유사도 % 아님
