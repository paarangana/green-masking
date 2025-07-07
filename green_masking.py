import cv2
import numpy as np

capture = cv2.VideoCapture(1)

while True:
    ret, frame = capture.read()
    if not ret:
        break
   
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower= np.array([35, 40, 40])
    upper= np.array([85, 255, 255])

    masking = cv2.inRange(hsv, lower, upper)
    green_mask = cv2.bitwise_and(frame, frame, mask=masking)

    cv2.imshow('webcam', frame)
    cv2.imshow('detecting green areas', masking)
    cv2.imshow('green areas',green_mask)

    if cv2.waitKey(1) & 0xFF == ord('w'):
        break

capture.release()
cv2.destroyAllWindows()

