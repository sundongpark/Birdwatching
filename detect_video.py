import cv2
import numpy as np
import sys
import yolo

size_list = [320, 416, 608]
filename = sys.argv[1]

fps = 30.0
cap = cv2.VideoCapture(filename)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("result.mp4", fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if (cv2.waitKey(1) & 0xFF == ord("q")):
        break
    frame = yolo.yolo(frame=frame, size=size_list[0], score_threshold=0.1, nms_threshold=0.3)
    cv2.imshow("Bird Watching", frame)
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()