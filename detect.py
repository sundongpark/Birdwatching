import cv2
import numpy as np
import sys
import yolo

size_list = [320, 416, 608]

#입력 인자 값을 구분하여 기본값을 설정
size, score_threshold, nms_threshold = size_list[2], 10, 50

if len(sys.argv) == 5:
    size = int(sys.argv[2])
    score_threshold = int(sys.argv[3])
    nms_threshold = int(sys.argv[4])
elif len(sys.argv) == 4:
    size = int(sys.argv[2])
    score_threshold = int(sys.argv[3])
elif len(sys.argv) == 3:
    size = int(sys.argv[2])
elif len(sys.argv) < 2:
    print("입력이 올바르지 않습니다.")
    exit()

if size > 2048:
    print("Score Threshold가 너무 큽니다.")
    exit()
if score_threshold > 100:
    print("Score Threshold가 너무 큽니다.")
    exit()
if nms_threshold > 100:
    print("NMS Threshold가 너무 큽니다.")
    exit()

filename = sys.argv[1]
title = "Bird Watching"
img = cv2.imread(filename)

#트랙바의 변화에 따라 YOLO 프레임이 변화
def detect(x):
    img = cv2.imread(filename)
    size = cv2.getTrackbarPos("Size", title)
    size -= size % 32                           #Size는 32의 배수
    score_threshold = cv2.getTrackbarPos("Score Threshold", title) / 100.0
    nms_threshold = cv2.getTrackbarPos("NMS Threshold", title) / 100.0
    #YOLO 처리한 프레임을 가져온다
    frame = yolo.yolo(frame = img, size = size, score_threshold = score_threshold, nms_threshold = nms_threshold)
    cv2.imshow(title, frame)            #결과를 보여줌
    cv2.imwrite("result.jpg", frame)    #result.jpg로 결과를 저장

cv2.namedWindow(title)
#size가 초깃값이고 2048이 최댓값인 트랙바 생성
cv2.createTrackbar("Size", title, size, 2048, detect)
#score_threshold가 초깃값이고 100이 최댓값인 트랙바 생성
cv2.createTrackbar("Score Threshold", title, score_threshold, 100, detect)
#nms_threshold가 초깃값이고 100이 최댓값인 트랙바 생성
cv2.createTrackbar("NMS Threshold", title, nms_threshold, 100, detect)

cv2.waitKey(0)
cv2.destroyAllWindows()