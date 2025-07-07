# Vegetation Detection using Colour Masking 
This project uses OpenCV and NumPy to detect healthy vegetation in real time using a webcam. It identifies green regions (indicating healthy plants) and creates a mask where green areas are white and everything else is black.

Features:
- Real time webcam input
- Green colour detection using HSV masking
- Binary mask highlighting healthy vegetation
- Blurring to reduce noise

Main functions used:
- cv2.VideoCapture(): Accesses webcam feed
- cv2.cvtColor(): Converts frame from BGR to HSV color space
- cv2.inRange(): Creates binary mask for green color detection
- cv2.bitwise_and(): Applies the mask to show green areas only
- cv2.GaussianBlur(): Smoothens image to reduce noise

