import cv2
import numpy as np
import os
import glob

def evaluate_and_analyze_health(data_dir):
    # 1. Define the two distinct color ranges
    # Healthy Green Range
    lower_healthy = np.array([35, 40, 40])
    upper_healthy = np.array([85, 255, 255])
    
    # Unhealthy Yellow/Brown Range
    lower_unhealthy = np.array([10, 40, 40])
    upper_unhealthy = np.array([34, 255, 255])
    
    img_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    if not img_paths:
        print(f"No .jpg images found in '{data_dir}'. Check your path!")
        return

    total_accuracy = 0
    total_iou = 0
    total_health_score = 0
    count = 0

    print(f"Starting multi-mask evaluation on {len(img_paths)} field images...\n")
    print(f"{'Image Name':<25} | {'Accuracy (%)':<14} | {'IoU (%)':<10} | {'Health Score (%)':<15}")
    print("-" * 74)

    for img_path in sorted(img_paths):
        filename = os.path.basename(img_path)
        base_name, _ = os.path.splitext(filename)
        gt_path = os.path.join(data_dir, f"{base_name}_mask.png")
        
        if not os.path.exists(gt_path):
            continue
            
        img = cv2.imread(img_path)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        _, gt_binary = cv2.threshold(gt_mask, 0, 255, cv2.THRESH_BINARY)

        # 2. Convert to HSV and generate BOTH masks
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        healthy_mask = cv2.inRange(hsv, lower_healthy, upper_healthy)
        unhealthy_mask = cv2.inRange(hsv, lower_unhealthy, upper_unhealthy)

        # 3. Combine masks for dataset evaluation (Total Predicted Vegetation)
        # cv2.bitwise_or combines the white pixels from both masks
        pred_mask_combined = cv2.bitwise_or(healthy_mask, unhealthy_mask)

        # 4. Calculate Accuracy & IoU using the combined mask
        total_pixels = pred_mask_combined.size 
        correct_pixels = np.sum(pred_mask_combined == gt_binary)
        accuracy = (correct_pixels / total_pixels) * 100
        
        intersection = np.sum(np.logical_and(pred_mask_combined == 255, gt_binary == 255))
        union = np.sum(np.logical_or(pred_mask_combined == 255, gt_binary == 255))
        iou = (intersection / union) * 100 if union != 0 else 100.0

        # 5. Calculate the Vegetation Health Score
        healthy_pixels = np.sum(healthy_mask == 255)
        unhealthy_pixels = np.sum(unhealthy_mask == 255)
        total_veg_pixels = healthy_pixels + unhealthy_pixels
        
        if total_veg_pixels > 0:
            health_score = (healthy_pixels / total_veg_pixels) * 100
        else:
            health_score = 0.0  # No vegetation detected at all

        # Accumulate metrics
        total_accuracy += accuracy
        total_iou += iou
        total_health_score += health_score
        count += 1

        print(f"{filename[:23]:<25} | {accuracy:<14.2f} | {iou:<10.2f} | {health_score:<15.2f}")

    if count > 0:
        print("-" * 74)
        print(f"{'OVERALL AVERAGE':<25} | {total_accuracy/count:<14.2f} | {total_iou/count:<10.2f} | {total_health_score/count:<15.2f}")
    else:
        print("No matching image/mask pairs could be processed.")

if __name__ == "__main__":
    TARGET_FOLDER = "test" 
    make_the_python_file = evaluate_and_analyze_health(TARGET_FOLDER)