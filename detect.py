import cv2
import numpy as np
import sys
import yolo

size_list = [320, 416, 608]
filename = sys.argv[1]
title = "Bird Watching"
img = cv2.imread(filename)

def detect(x):
    img = cv2.imread(filename)
    size = cv2.getTrackbarPos("Size", title)
    size -= size % 32
    score_threshold = cv2.getTrackbarPos("Score Threshold", title) / 100.0
    nms_threshold = cv2.getTrackbarPos("NMS Threshold", title) / 100.0
    frame = yolo.yolo(frame = img, size = size, score_threshold = score_threshold, nms_threshold = nms_threshold)
    cv2.imshow(title, frame)
    cv2.imwrite("result.jpg", frame)

cv2.namedWindow(title)
cv2.createTrackbar("Size", title, size_list[2], 4096, detect)
cv2.createTrackbar("Score Threshold", title, 10, 100, detect)
cv2.createTrackbar("NMS Threshold", title, 50, 100, detect)

cv2.waitKey(0)
cv2.destroyAllWindows()