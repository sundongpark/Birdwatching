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
size -= size % 32       #Size는 32의 배수
score_threshold /= 100.0
nms_threshold /= 100.0
cap = cv2.VideoCapture(filename)

#결과 저장을 위한 변수들
fps = 30.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("result.mp4", fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

#비디오의 프레임이 끝날 때까지 반복
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if (cv2.waitKey(1) & 0xFF == ord("q")):
        break
    frame = yolo.yolo(frame = frame, size = size, score_threshold = score_threshold, nms_threshold = nms_threshold)
    cv2.imshow("Bird Watching", frame)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()