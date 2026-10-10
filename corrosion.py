# =========================================================
# CORROSION DETECTION & SEVERITY ASSESSMENT
# =========================================================

import cv2
import numpy as np
import pandas as pd
import os

# =========================================================
# 1. CONVERT COLOR SPACE
# =========================================================
def convert_color_spaces(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv


# =========================================================
# 2. DETECT CORROSION (HSV)
# =========================================================
def detect_corrosion(hsv):
    lower = np.array([5, 100, 50])     # rust color range
    upper = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    return mask


# =========================================================
# 3. CLEAN MASK
# =========================================================
def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


# =========================================================
# 4. REMOVE FALSE DETECTIONS
# =========================================================
def remove_noise(hsv, mask):
    # Remove green (algae)
    green_lower = np.array([35, 50, 50])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    mask[green_mask > 0] = 0

    # Remove shadows
    value = hsv[:, :, 2]
    mask[value < 50] = 0

    return mask


# =========================================================
# 5. AREA CALCULATION
# =========================================================
def calculate_area(mask):
    corrosion_pixels = np.sum(mask > 0)
    total_pixels = mask.shape[0] * mask.shape[1]

    percent = (corrosion_pixels / total_pixels) * 100
    return corrosion_pixels, percent


# =========================================================
# 6. SEVERITY
# =========================================================
def severity_score(corrosion_percent, crack_length):
    return 0.6 * corrosion_percent + 0.4 * crack_length


def classify(score):
    if score < 20:
        return "Low"
    elif score < 50:
        return "Medium"
    else:
        return "High"


# =========================================================
# 7. PROCESS SINGLE IMAGE
# =========================================================
def process_image(path, crack_length=10):
    image = cv2.imread(path)

    if image is None:
        print("❌ Error loading:", path)
        return None

    hsv = convert_color_spaces(image)

    mask = detect_corrosion(hsv)
    mask = clean_mask(mask)
    mask = remove_noise(hsv, mask)

    pixels, percent = calculate_area(mask)

    score = severity_score(percent, crack_length)
    level = classify(score)

    return image, mask, pixels, percent, score, level


# =========================================================
# 8. PROCESS DATASET
# =========================================================
def process_dataset(folder):
    results = []

    os.makedirs("output", exist_ok=True)

    for file in os.listdir(folder):
        if file.endswith((".jpg", ".png", ".jpeg")):

            path = os.path.join(folder, file)

            data = process_image(path)

            if data is None:
                continue

            image, mask, pixels, percent, score, level = data

            # Save mask
            cv2.imwrite(f"output/mask_{file}", mask)

            # Save overlay
            overlay = cv2.bitwise_and(image, image, mask=mask)
            cv2.imwrite(f"output/overlay_{file}", overlay)

            # Save results
            results.append({
                "Image": file,
                "Corrosion %": round(percent, 2),
                "Severity Score": round(score, 2),
                "Level": level
            })

            print(f"✅ {file} → {level}")

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv("output/results.csv", index=False)

    print("\n🎯 DONE! Check 'output' folder")


# =========================================================
# 9. MAIN
# =========================================================
if __name__ == "__main__":
    dataset_folder = "dataset_images"   # your image folder
    process_dataset(dataset_folder)
