import cv2
import numpy as np

def yolo(frame, size, score_threshold, nms_threshold):
    net = cv2.dnn.readNet(f"yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255 , (size, size), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores) #14
            confidence = scores[class_id]

            if confidence > 0.1:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    print(f"boxes: {boxes}")
    print(f"confidences: {confidences}")

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=score_threshold, nms_threshold=nms_threshold)
    
    print(f"indexes: ", end='')
    for index in indexes:
        print(index, end=' ')

    print("\n\n============================== classes ==============================")

    birds = 0
    for i in range(len(boxes)):
        if i in indexes:
            if classes[class_ids[i]] == "bird":
                birds += 1
                x, y, w, h = boxes[i]
                class_name = classes[class_ids[i]]
                label = f"{class_name} {confidences[i]:.2f}"
                color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                print(f"[{class_name}({i})] conf: {confidences[i]} / x: {x} / y: {y} / width: {w} / height: {h}")

    cv2.rectangle(frame, (0, 0), (130, 30), (255, 255, 255), -1)
    cv2.putText(frame, f"birds: {birds}", (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    return frame

classes = ["person", "bicycle", "car", "motorcycle",
           "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
           "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
           "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
           "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
           "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
           "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
           "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


size_list = [320, 416, 608]
filename = "bird.mp4"
video = True

if video:
    cap = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap.read()
        if (cv2.waitKey(1) & 0xFF == ord("q")):
            break
        frame = yolo(frame=frame, size=size_list[0], score_threshold=0.1, nms_threshold=0.3)
        cv2.imshow("Frame", frame)
    cap.release()
else:
    frame = cv2.imread(filename)
    frame = yolo(frame=frame, size=size_list[2], score_threshold=0.1, nms_threshold=0.3)
    cv2.imshow("Bird Detection", frame)
    cv2.imwrite("result.jpg", frame)
    cv2.waitKey(0)
cv2.destroyAllWindows()
