import cv2
import numpy as np
import sys
import yolo

size_list = [320, 416, 608]
filename = sys.argv[1]

frame = cv2.imread(filename)
frame = yolo.yolo(frame=frame, size=size_list[2], score_threshold=0.1, nms_threshold=0.3)
cv2.imshow("Bird Watching", frame)
cv2.imwrite("result.jpg", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
