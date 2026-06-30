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



## Performance Optimization Study (Dataset Benchmarking)

To rigorously test the edge cases of our rule-based HSV masking, we evaluated the pipeline against a standardized agricultural segmentation dataset using **Pixel Accuracy** and **Intersection over Union (IoU)** metrics.

### Iteration 1: Strict Green Filter
* **Pixel Accuracy:** 77.01%
* **Intersection over Union (IoU):** 32.71%
* *Finding:* High baseline accuracy but low IoU because the mask completely ignored parched or nutrient-stressed crops, failing to overlap with the full vegetation footprint.

### Iteration 2: Dual-Mask System (Healthy Green + Stressed Yellow/Brown)
* **Pixel Accuracy:** 55.47%
* **Intersection over Union (IoU):** 47.85%
* **Average Vegetation Health Score:** 22.25%
* *Finding:* Broadening the color profile captured a significantly higher proportion of crop areas, raising the IoU by **15.14%**. However, background soil shares a near-identical HSV profile with dry, yellowing leaves, creating false positives that dropped the overall pixel accuracy.

### 🏃 How to Run the Dataset Evaluation
1. Navigate to the analysis directory: `cd dataset_analysis`
2. Run the script: `python green_masking.py`
